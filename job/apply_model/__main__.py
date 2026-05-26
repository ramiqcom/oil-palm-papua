from concurrent.futures import ThreadPoolExecutor
from math import ceil
from subprocess import check_call
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import rasterio as rio
from keras.models import load_model
from rasterio import transform, windows
from shapely.geometry import box

from job.utils import MAX_WORKERS, logger

MAX_WORKERS = MAX_WORKERS
CPU_PER_PROCESS = 8
PROCESS_COUNT = int(MAX_WORKERS / CPU_PER_PROCESS)

INPUT_PREFIX = "/usr/src/app/input"
OUTPUT_PREFIX = "/usr/src/app/output"
MODEL_NAME = "unet_chm_v1_128x128_sampleCount6472_1779751763"
MODEL_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}.keras"
PREDICTION_PATH = f"{OUTPUT_PREFIX}/prediction_oil_palm_{MODEL_NAME}.tif"
REGION_NAME = "papua_selatan"
VERSION = "v1_unet"
BATCH_SIZE = 64
ROI = f"{INPUT_PREFIX}/roi/papua_selatan_oil_palm_bounds.fgb"

YEARS_DATA = [
    dict(
        year=2020,
        ms_path=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2020-01-01_2020-12-31_30m.tif",
    ),
]

logger.info("Load model")
model = load_model(MODEL_PATH)

resolution = 30
size = 128
prediction_size = resolution * size
degree_distance = prediction_size / 111_000

gdf = gpd.read_file(ROI)
BBOX = tuple(gdf.total_bounds)
union = gdf.union_all()
ORIGINAL_HEIGHT = abs(BBOX[1] - BBOX[3]) * 111_000
ORIGINAL_WIDTH = abs(BBOX[0] - BBOX[2]) * 111_000
PIXEL_HEIGHT = int(ORIGINAL_HEIGHT / resolution)
PIXEL_WIDTH = int(ORIGINAL_WIDTH / resolution)
count_x = ceil(ORIGINAL_WIDTH / prediction_size)
count_y = ceil(ORIGINAL_HEIGHT / prediction_size)

all_grids = []
for x in range(count_x):
    min_x = BBOX[0] + (x * degree_distance)
    max_x = min_x + degree_distance

    if max_x > BBOX[2]:
        max_x = BBOX[2]

    for y in range(count_y):
        min_y = BBOX[1] + (y * degree_distance)
        max_y = min_y + degree_distance

        if max_y > BBOX[3]:
            max_y = BBOX[3]

        polygon = box(min_x, min_y, max_x, max_y)

        if union.contains(polygon):
            all_grids.append(
                dict(
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    geometry=polygon,
                )
            )


def rescale(array: np.ndarray, src: tuple[float, float]):
    return ((array - src[0]) / (src[1] - src[0])).astype("float32")


def predict_image(bbox: tuple[float, float, float, float]):
    folder = TemporaryDirectory(delete=False)
    o_temp = f"{folder.name}/output.tif"

    with rio.open(YEARS_DATA[0]["ms_path"]) as src:
        logger.info(f"Load image {bbox}")
        window = windows.from_bounds(*bbox, transform=src.transform)
        profile = src.profile
        nir, swir1, swir2 = src.read(
            [4, 5, 6], out_shape=(prediction_size, prediction_size), window=window
        )
        batch = (
            np.dstack(
                [
                    rescale(nir, (1000, 4000)) * 255,
                    rescale(swir1, (500, 3000)) * 255,
                    rescale(swir2, (250, 2000)) * 255,
                ]
            ).astype("uint8")
            / 255
        ).astype("float32")

        batch = np.stack([batch])

        # Predict
        logger.info(f"Predict {bbox}")
        predicted = model.predict(batch)[0, :, :, 0]

        del batch

        predicted = np.round(predicted)

        profile["driver"] = "COG"
        profile["count"] = 1
        profile["nodata"] = 0
        profile["dtype"] = "uint8"
        profile["resampling"] = "nearest"
        profile["transform"] = transform.from_bounds(
            *bbox, height=prediction_size, width=prediction_size
        )
        profile["height"] = prediction_size
        profile["width"] = prediction_size

        logger.info("Save the whole data")
        with rio.open(o_temp, "w", **profile) as o:
            o.write(predicted, 1)
            o.descriptions = tuple(["oil_palm"])

        return o_temp


with ThreadPoolExecutor(2) as executor:
    jobs = [
        executor.submit(
            predict_image,
            bbox=(dict["min_x"], dict["min_y"], dict["max_x"], dict["max_y"]),
        )
        for dict in all_grids
    ]
    results = [job.result() for job in jobs]

with TemporaryDirectory() as folder:
    logger.info("Mosaic all")
    mosaic = PREDICTION_PATH
    check_call(
        f"""gdal raster mosaic \
            -f COG \
            --co="COMPRESS=ZSTD" \
            --co="STATISTICS=YES" \
            --co="OVERVIEWS=IGNORE_EXISTING" \
            --co="OVERVIEW_RESAMPLING=NEAREST" \
            --co="RESAMPLING=NEAREST" \
            {" ".join(results)} \
            {mosaic} \
            --overwrite \
        """,
        shell=True,
    )
