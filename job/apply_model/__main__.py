import numpy as np
import rasterio as rio
from keras.models import load_model

from job.utils import MAX_WORKERS, logger

MAX_WORKERS = MAX_WORKERS
CPU_PER_PROCESS = 8
PROCESS_COUNT = int(MAX_WORKERS / CPU_PER_PROCESS)

INPUT_PREFIX = "/usr/src/app/input"
OUTPUT_PREFIX = "/usr/src/app/output"
MODEL_NAME = "unet_chm_v1_128x128_sampleCount6472_1779627865"
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


def rescale(array: np.ndarray, src: tuple[float, float]):
    return ((array - src[0]) / (src[1] - src[0])).astype("float32")


prediction_shape = 3840

with rio.open(YEARS_DATA[0]["ms_path"]) as src:
    profile = src.profile
    nir, swir1, swir2 = src.read(
        [4, 5, 6], out_shape=(prediction_shape, prediction_shape)
    )
    composite = np.dstack(
        [
            rescale(nir, (1000, 4000)),
            rescale(swir1, (500, 3000)),
            rescale(swir2, (250, 2000)),
        ]
    )

    batch = np.stack([composite])

    logger.info("Load model")
    model = load_model(MODEL_PATH)

    # Predict
    logger.info("Predict the whole data")
    predicted = model.predict(batch)[0, :, :, 0]
    predicted = np.round(predicted)

    profile["driver"] = "COG"
    profile["count"] = 1
    profile["nodata"] = 0
    profile["dtype"] = "uint8"
    profile["resampling"] = "nearest"

    logger.info("Save the whole data")
    with rio.open(PREDICTION_PATH, "w", **profile) as o:
        o.write(predicted, 1)
        o.descriptions = tuple(["oil_palm"])
