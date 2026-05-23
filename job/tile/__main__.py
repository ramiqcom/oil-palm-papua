from concurrent.futures import ThreadPoolExecutor
from math import ceil
from os import mkdir, remove
from subprocess import check_call

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import box

from ..utils import MAX_WORKERS, logger

MAX_WORKERS = MAX_WORKERS
CPU_PER_PROCESS = 8
PROCESS_COUNT = int(MAX_WORKERS / CPU_PER_PROCESS)

RANDOM_STATE = 1
TEST_RATIO = 0.3
VERSION = "v1_unet"
REGION_NAME = "papua_selatan"
INPUT_PREFIX = "/usr/src/app/input"


ROI = f"{INPUT_PREFIX}/roi/papua_selatan_oil_palm_bounds.fgb"

YEARS_DATA = [
    dict(
        year=2020,
        ms_path=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2020-01-01_2020-12-31_30m.tif",
        label=f"{INPUT_PREFIX}/label/sample_raster.tif"
    ),
]

LABEL = "OILPALM"

roi_df = gpd.read_file(ROI)
union = roi_df.union_all()
BBOX = tuple(roi_df.total_bounds)
ORIGINAL_HEIGHT = abs(BBOX[1] - BBOX[3]) * 111_000
ORIGINAL_WIDTH = abs(BBOX[0] - BBOX[2]) * 111_000
PIXEL_HEIGHT = int(ORIGINAL_HEIGHT / 30)
PIXEL_WIDTH = int(ORIGINAL_WIDTH / 30)

# in meter
SCALES = [3000, 6000, 12000]
BUFFERS = [0.1, 0.2, 0.3]
TARGET_SIZE = 128
FLIPS = [0, 1]
ROTATIONS = [0, 1, 2, 3]

# output
OUTPUT_PREFIX = "/usr/src/app/output"
TILE_PREFIX = f"{OUTPUT_PREFIX}/tile_images"
IMAGE_PREFIX = f"{TILE_PREFIX}/image"
LABEL_PREFIX = f"{TILE_PREFIX}/label"

try:
    mkdir(TILE_PREFIX)
except Exception:
    None
try:
    mkdir(IMAGE_PREFIX)
except Exception:
    None
try:
    mkdir(LABEL_PREFIX)
except Exception:
    None

# generate grids
logger.info("Generate grids")
all_grids = []
for scale in SCALES:
    count_x = ceil(ORIGINAL_WIDTH / scale)
    count_y = ceil(ORIGINAL_HEIGHT / scale)
    distance = scale / 111_000
    for buf in BUFFERS:
        for x in range(count_x):
            min_x = BBOX[0] + (x * distance)
            max_x = min_x + distance
            for y in range(count_y):
                min_y = BBOX[1] + (y * distance)
                max_y = min_y + distance

                polygon = box(min_x, min_y, max_x, max_y)
                polygon = box(*polygon.buffer(distance * buf).bounds)

                if union.contains(polygon):
                    all_grids.append(
                        dict(
                            min_x=min_x,
                            max_x=max_x,
                            min_y=min_y,
                            max_y=max_y,
                            geometry=polygon,
                            scale=scale,
                            buffer=buf,
                        )
                    )
all_grids = gpd.GeoDataFrame(all_grids, crs="EPSG:4326")
grids_output = f"{OUTPUT_PREFIX}/grids.geojson"
all_grids.to_file(grids_output)

def grid_to_image(all_grids, index):
    logger.info(f"Run grid {index + 1}")

    grid = all_grids.iloc[index]
    min_x = grid["min_x"]
    min_y = grid["min_y"]
    max_x = grid["max_x"]
    max_y = grid["max_y"]
    output_image = f"{IMAGE_PREFIX}/{index + 1}_IMAGE.tif"
    output_label = f"{LABEL_PREFIX}/{index + 1}_LABEL.tif"

    check_call(
        f"""gdal raster pipeline \
            ! read {YEARS_DATA[0]["ms_path"]} \
            ! select --band=4,5,6 \
            ! reproject \
                --bbox={min_x},{min_y},{max_x},{max_y} \
                --size={TARGET_SIZE},{TARGET_SIZE} \
            ! write -f COG {output_image} --overwrite \
        """,
        shell=True,
    )

    with rio.open(output_image) as o:
        image = o.read(1)
        if np.sum((image == 0) * 1) > 0:
            remove(output_image)
            raise Exception(f"Grid no {index + 1} image contain no data")

    check_call(
        f"""gdal raster pipeline \
            ! read {output_image} \
            ! scale -b 1 --src-min=1000 --src-max=4000 --dst-min=1 --dst-max=255 \
            ! scale -b 2 --src-min=500 --src-max=3000 --dst-min=1 --dst-max=255 \
            ! scale -b 3 --src-min=250 --src-max=2000 --dst-min=1 --dst-max=255 \
            ! set-type --ot=Byte \
            ! write -f COG {output_image} --overwrite
        """,
        shell=True,
    )

    check_call(
        f"""gdal raster pipeline \
            ! read {YEARS_DATA[0]["label"]} \
            ! reproject \
                --bbox={min_x},{min_y},{max_x},{max_y} \
                --size={TARGET_SIZE},{TARGET_SIZE} \
            ! write -f COG {output_label} --overwrite \
        """,
        shell=True,
    )

    with rio.open(output_label) as o:
        image = o.read(1)
        if np.sum((image == 1) * 1) < 1:
            remove(output_label)
            remove(output_image)
            raise Exception(f"Grid no {index + 1} label contain no oil palm")
        else:
             with rio.open(output_image) as o:
                image = o.read()
                image_profile = o.profile
                image_profile["driver"] = "COG"

             with rio.open(output_label) as o:
                label = o.read()
                label_profile = o.profile
                label_profile["driver"] = "COG"


                for flip in FLIPS:
                    image_flip = np.flip(image, 2) if (flip != 0) else image
                    label_flip = np.flip(label, 2) if (flip != 0) else label

                    for rot in ROTATIONS:
                        if  not ((flip == 0) and (rot == 0)):
                            image_rot = np.rot90(image_flip, rot, (1, 2)) if (rot != 0) else image_flip
                            label_rot = np.rot90(label_flip, rot, (1, 2)) if (rot != 0) else label_flip

                            with rio.open(f"{IMAGE_PREFIX}/{index + 1}F{flip}R{rot}_IMAGE.tif", "w", **image_profile) as o:
                                o.write(image_rot)

                            with rio.open(f"{LABEL_PREFIX}/{index + 1}F{flip}R{rot}_LABEL.tif", "w", **label_profile) as o:
                                o.write(label_rot)


with ThreadPoolExecutor(MAX_WORKERS) as executor:
    jobs = [
        executor.submit(grid_to_image, all_grids, index)
        for index in range(len(all_grids)) if index >= 570
    ]
    for job in jobs:
        try:
            job.result()
        except Exception as e:
            logger.info(f"Error: {e}")
