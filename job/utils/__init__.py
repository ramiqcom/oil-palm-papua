import json
import math
from concurrent.futures import ThreadPoolExecutor
from logging import INFO, basicConfig, getLogger
from os import cpu_count, remove
from subprocess import check_call, check_output
from tempfile import TemporaryDirectory
from typing import Callable

import geopandas as gpd
from pystac_client import Client
from shapely import intersects
from shapely.geometry import box

basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = getLogger(__name__)

# Base parameter
MAX_WORKERS = int(cpu_count() or 1)
CRS = "EPSG:4326"
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
NASA_STAC_LPCLOUD = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
NASA_STAC_ORNLCLOUD = "https://cmr.earthdata.nasa.gov/stac/ORNL_CLOUD"
COPERNICUS_STAC = "https://stac.dataspace.copernicus.eu/v1"
USGS_STAC = "https://landsatlook.usgs.gov/stac-server"

# DEM collection
DEM_COLLECTION_PC = "nasadem"
DEM_COLLECTION_EARTHDATA = "NASADEM_HGT_001"
DEM_BAND = "elevation"


# Function to get token for copernicus stac
def get_copernicus_token():
    token_dict = json.loads(
        check_output(
            """curl \
                -d client_id=cdse-public \
                -d username=$COPERNICUS_EMAIL \
                -d password=$COPERNICUS_PASSWORD \
                -d grant_type=password \
                https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
            """,
            shell=True,
            text=True,
        )
    )
    return token_dict["access_token"]


# Function filter STAC assets with GDAL
def filter_assets(
    collection: str,
    bbox: tuple[float, float, float, float],
    date: tuple[str, str] | None = None,
    stac_api: str = PLANETARY_COMPUTER_STAC,
):
    client = Client.open(stac_api)
    search = client.search(collections=[collection], bbox=bbox, datetime=date)
    features = [item for item in search.items_as_dicts()]
    logger.info(f"Found {len(features)} features")
    return features


# Function to download NASADEM HGT
def download_nasadem_hgt(feat: dict, folder: str):
    assets = feat["assets"]
    keys = assets.keys()
    key = [key for key in keys if key.startswith("001")][0]
    href = assets[key]["href"]
    name = href.split("_")[-1].split(".zip")[0]
    output_path = f"{folder}/{name}.zip"
    check_call(
        f"curl -o {output_path} -L -b .urs_cookies -c .urs_cookies --netrc-file /usr/src/app/.netrc {href}",
        shell=True,
    )
    return f"/vsizip/{output_path}/{name}.hgt"


# function to generate dem
def generate_dem(
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    source: str = "planetary_computer",
):
    logger.info("Generate DEM data")
    if source == "planetary_computer":
        features_dem = filter_assets(collection=DEM_COLLECTION_PC, bbox=bbox)
        dem = mosaic_images(
            images=[
                f"/vsicurl?pc_url_signing=yes&pc_collection={DEM_COLLECTION_PC}&url={feat['assets'][DEM_BAND]['href']}"
                for feat in features_dem
            ],
            reproject_param=f"-d EPSG:4326 --bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} --bbox-crs=EPSG:4326 --size={shape[0]},{shape[1]}",
        )
    elif source == "earthdata":
        with TemporaryDirectory() as folder:
            features_dem = filter_assets(
                collection=DEM_COLLECTION_EARTHDATA,
                bbox=bbox,
                stac_api=NASA_STAC_LPCLOUD,
            )
            links = []
            with ThreadPoolExecutor(MAX_WORKERS) as executor:
                jobs = [
                    executor.submit(download_nasadem_hgt, feat, folder)
                    for feat in features_dem
                ]
                links = [job.result() for job in jobs]
            dem = mosaic_images(
                images=links,
                reproject_param=f"-d EPSG:4326 --bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} --bbox-crs=EPSG:4326 --size={shape[0]},{shape[1]}",
            )
    return dem


# Function to generate terrain data
def generate_terrain(
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    source: str = "planetary_computer",
):
    temp_folder = TemporaryDirectory(delete=False)

    logger.info("Generate DEM data")
    dem = generate_dem(bbox=bbox, shape=shape, source=source)

    logger.info("Generate slope and aspect")
    with ThreadPoolExecutor(2) as executor:
        slope = f"{temp_folder.name}/slope.tif"
        aspect = f"{temp_folder.name}/aspect.tif"
        jobs = [
            executor.submit(
                check_call,
                f"gdal raster slope {dem} {slope}",
                shell=True,
            ),
            executor.submit(
                check_call,
                f"gdal raster aspect {dem} {aspect}",
                shell=True,
            ),
        ]
        for job in jobs:
            try:
                job.result()
            except Exception as e:
                logger.info(f"Error: {e}")

    logger.info("Generate slope and aspect derivatives")
    terrain_d = f"{temp_folder.name}/terrain_d.tif"
    check_call(
        f"""gdal raster calc \
            -i "S={slope}" \
            -i "A={aspect}" \
            -o {terrain_d} \
            --calc="S / 180 * _pi" \
            --calc="sin(S / 180 * _pi)" \
            --calc="cos(S / 180 * _pi)" \
            --calc="A / 180 * _pi" \
            --calc="sin(A / 180 * _pi)" \
            --calc="cos(A / 180 * _pi)" \
            --propagate-nodata \
            --nodata=NaN \
            --ot=Float32 \
            --co="COMPRESS=ZSTD"
        """,
        shell=True,
    )

    return slope, aspect, terrain_d


# Function to mosaic image
def mosaic_images(images: list[str], reproject_param: str):
    temp_folder = TemporaryDirectory(delete=False)
    clipped = f"{temp_folder.name}/mosaic.tif"
    output_param = """--co="COMPRESS=ZSTD" """

    try:
        list_file = f"{temp_folder.name}/list_file.txt"
        with open(list_file, "w") as file:
            file.write("\n".join(images))

        # Mosaic image if it list of images
        check_call(
            f"""gdal raster pipeline \
                ! mosaic --resolution=average -i @{list_file} \
                ! reproject {reproject_param} \
                ! write {output_param} {clipped}""",
            shell=True,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info("Will reproject first before mosaic")

        with ThreadPoolExecutor(MAX_WORKERS) as executor:
            jobs = []
            for index in range(len(images)):
                jobs.append(
                    executor.submit(
                        check_call,
                        f"""gdal raster reproject \
                            {reproject_param} \
                            {output_param} \
                            "{images[index]}" \
                            {temp_folder.name}/images_{index}.tif""",
                        shell=True,
                    )
                )

            images_reproject = []
            for index in range(len(images)):
                try:
                    jobs[index].result()
                    images_reproject.append(f"{temp_folder.name}/images_{index}.tif")
                except Exception as e:
                    logger.info(f"Error: {e}")

        list_file = f"{temp_folder.name}/list_file_reproject.txt"
        with open(list_file, "w") as file:
            file.write("\n".join(images_reproject))

        # Mosaic the dataset
        logger.info("Mosaic the reprojected dataset")
        check_call(
            f"""gdal raster pipeline \
                ! mosaic --resolution=average -i @{list_file} \
                ! reproject {reproject_param} \
                ! write {output_param} {clipped}""",
            shell=True,
        )

    return clipped


# Function to create median per band
def aggregate_median(
    images: list[str],
    bbox: tuple[float, float, float, float],
    shape: tuple[float, float],
):
    temp_folder = TemporaryDirectory(delete=False)

    paths_file = f"{temp_folder.name}/images.txt"
    with open(paths_file, "w") as file:
        file.write("\n".join(images))

    logger.info("Apply median composite")
    median = f"{temp_folder.name}/median.tif"
    check_call(
        f"""gdal raster pipeline \
            ! mosaic \
              -i @{paths_file} \
              --bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} \
              --pixel-function=median \
              --resolution=average \
            ! reproject -d EPSG:4326 --size={shape[0]},{shape[1]} --bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} --bbox-crs=EPSG:4326 \
            ! write --co="COMPRESS=ZSTD" {median}
        """,
        shell=True,
    )

    logger.info("Remove images after median composite")
    for path in images:
        remove(path)

    return median


# Function to process vector input
def process_roi(
    path: str,
    resolution: float = 30,
    sql_where: str | None = None,
    rasterize: bool = False,
):
    if path.startswith("http"):
        path = f"/vsicurl/{path}"

    temp_folder = TemporaryDirectory(delete=False)
    fix_vector = f"{temp_folder.name}/input_vector.fgb"
    check_call(
        f"""gdal pipeline \
            ! read {path} \
            {f'''! filter --where="{sql_where}" ''' if (sql_where is not None) and (sql_where != "") else ""} \
            ! set-geom-type --dim=XY \
            ! reproject -d EPSG:4326 \
            ! explode-collections \
            ! make-valid \
            ! materialize \
            {f"! rasterize --ot=Byte --init=0 --nodata=0 --burn=1 --resolution={resolution / 111_000},{resolution / 111_000} ! materialize ! polygonize ! materialize" if rasterize else ""} \
            ! explode-collections \
            ! make-valid \
            ! write -f FlatGeobuf --lco="SPATIAL_INDEX=YES" {fix_vector}
        """,
        shell=True,
    )

    # Read the vector info
    vector_info = json.loads(
        check_output(f"gdal vector info -f json {fix_vector}", shell=True)
    )

    # Get the bbox
    bbox = vector_info["layers"][0]["geometryFields"][0]["extent"]

    # Define width and height
    width = int(abs(bbox[0] - bbox[2]) * 111_000 / resolution)
    height = int(abs(bbox[1] - bbox[3]) * 111_000 / resolution)

    return fix_vector, bbox, (width, height)


def generate_grids(
    fix_vector: str, bbox: tuple[float, float, float, float], size: float = 1
):
    logger.info("Generate grids")
    temp_folder = TemporaryDirectory(delete=False)
    distance = size  # in degree
    min_x, min_y, max_x, max_y = bbox
    count_x = math.ceil(abs(min_x - max_x) / distance)
    count_y = math.ceil(abs(min_y - max_y) / distance)
    grids = []

    # Features geometry
    features = gpd.read_file(fix_vector)

    for x in range(count_x):
        x1 = min_x + (distance * x)
        x2 = x1 + distance
        for y in range(count_y):
            y1 = min_y + (distance * y)
            y2 = y1 + distance
            polygon = box(x1, y1, x2, y2)

            features_box = features.cx[x1:x2, y1:y2].union_all()

            if intersects(polygon, features_box):
                grids.append(
                    dict(
                        grid_id=f"Y{y:03d}_X{x:03d}",
                        min_x=x1,
                        min_y=y1,
                        max_x=x2,
                        max_y=y2,
                        geometry=polygon,
                    )
                )

    grids_path = f"{temp_folder.name}/grids.fgb"
    grids = gpd.GeoDataFrame(grids, crs="EPSG:4326")
    grids.to_file(grids_path, **{"SPATIAL_INDEX": "YES"})

    return grids_path


def process_grid(
    grids: str,
    grid_id: str,
    resolution: float,
    composite_function: Callable,
    date: tuple[str, str] | None = None,
):
    logger.info(f"Process grid {grid_id}")
    grid_vector, bbox, shape = process_roi(
        path=grids, resolution=resolution, sql_where=f"grid_id = '{grid_id}'"
    )

    if date is not None:
        composite_result = composite_function(
            bbox=bbox,
            shape=shape,
            date=date,
        )
    else:
        composite_result = composite_function(
            bbox=bbox,
            shape=shape,
        )

    return composite_result


def mosaic_grids(
    images: list[str],
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    vector_for_clip: str,
):
    logger.info("Mosaic all grids result")
    temp_folder = TemporaryDirectory(delete=False)

    # Save all images into a text file
    image_list = f"{temp_folder.name}/image_list.txt"
    with open(image_list, "w") as file:
        file.write("\n".join(images))

    # Mosaic final image
    mosaic = f"{temp_folder.name}/mosaic.tif"
    check_call(
        f"""gdal raster pipeline \
            ! mosaic --resolution=average -i @{image_list} \
            ! clip --like={vector_for_clip} --allow-bbox-outside-source \
            ! reproject -d EPSG:4326 --bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} --bbox-crs=EPSG:4326 --dst-nodata=0 --size={shape[0]},{shape[1]} \
            ! write -f COG --co="COMPRESS=ZSTD" --co="RESAMPLING=LANCZOS" --co="OVERVIEWS=IGNORE_EXISTING" --co="OVERVIEW_RESAMPLING=LANCZOS" --co="STATISTICS=YES" {mosaic}
        """,
        shell=True,
    )

    for path in images:
        remove(path)

    return mosaic
