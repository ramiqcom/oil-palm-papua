import json
from concurrent.futures import ThreadPoolExecutor
from os import mkdir
from shutil import copyfile
from subprocess import check_call
from tempfile import NamedTemporaryFile, TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from PIL import ImageColor
from rasterio import transform, windows
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRFClassifier

from job.utils import MAX_WORKERS, logger

INPUT_PREFIX = "/usr/src/app/input"
OUTPUT_PREFIX = "/usr/src/app/output"
ADM = f"{INPUT_PREFIX}/admin/indonesia_adm_level_0.tif"
ROI = f"{INPUT_PREFIX}/roi/papua_selatan_oil_palm_bounds.fgb"
DEM = f"{INPUT_PREFIX}/nasadem.tif"
BUCKET = "gee-ramadhan-s4g-bucket"
REGION_NAME = "papua_selatan"
SAMPLE_SIZE = 1_000_000

try:
    mkdir(f"{OUTPUT_PREFIX}/prediction_lc_klhk_s4g_v1")
except Exception:
    ""

LCS = [
    dict(
        year=2000,
        lc=f"{INPUT_PREFIX}/lc_klhk/lc_2000.tif",
    ),
    dict(
        year=2011,
        lc=f"{INPUT_PREFIX}/lc_klhk/lc_2011.tif",
    ),
    dict(
        year=2016,
        lc=f"{INPUT_PREFIX}/lc_klhk/lc_2016.tif",
    ),
    dict(
        year=2019,
        lc=f"{INPUT_PREFIX}/lc_klhk/lc_2019.tif",
    ),
    dict(
        year=2021,
        lc=f"{INPUT_PREFIX}/lc_klhk/lc_2021.tif",
    ),
]

EOS = [
    dict(
        year=2000,
        ms=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2000-01-01_2000-12-31_30m.tif",
    ),
    dict(
        year=2005,
        ms=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2005-01-01_2005-12-31_30m.tif",
    ),
    dict(
        year=2010,
        ms=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2009-07-01_2011-06-30_30m.tif",
    ),
    dict(
        year=2015,
        ms=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2015-01-01_2015-12-31_30m.tif",
    ),
    dict(
        year=2020,
        ms=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2020-01-01_2020-12-31_30m.tif",
    ),
    dict(
        year=2025,
        ms=f"{INPUT_PREFIX}/glad_ard/papua_selatan_oil_palm_GLAD_ARD_2024-07-01_2026-05-01_30m.tif",
    ),
]

INDICES = [
    dict(name="NDVI", band1="NIR", band2="RED"),
    dict(name="NDMI", band1="NIR", band2="SWIR1"),
    dict(name="NBR", band1="NIR", band2="SWIR2"),
    dict(name="NBR2", band1="SWIR1", band2="SWIR2"),
    dict(name="NDWI", band1="GREEN", band2="NIR"),
    dict(name="MNDWI", band1="GREEN", band2="SWIR1"),
    dict(name="MNDWI2", band1="GREEN", band2="SWIR2"),
]

LABEL_LC = "LC"
INDICES_BANDS = [data["name"] for data in INDICES]
MS_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
DEM_BANDS = ["DEM"]
BANDS = [
    *MS_BANDS,
    *DEM_BANDS,
]
PREDICTORS = [*BANDS, *INDICES_BANDS]
COLUMNS = [*BANDS, LABEL_LC]
TEST_RATIO = 0.3
RANDOM_STATE = 1
RESOLUTION = 30
BBOX = tuple(gpd.read_file(ROI).total_bounds)
HEIGHT = int(abs(BBOX[1] - BBOX[3]) * 111_000 / RESOLUTION)
WIDTH = int(abs(BBOX[0] - BBOX[2]) * 111_000 / RESOLUTION)
TARGET_TRANSFORM = transform.from_bounds(*BBOX, WIDTH, HEIGHT)

with open(f"{INPUT_PREFIX}/lc.json") as file:
    lc_info = json.load(file)
    values = lc_info["values"]

    s4g_values_conversion = lc_info["s4g_values_conversion"]

    s4g_values = lc_info["s4g_values"]
    s4g_labels = lc_info["s4g_labels"]
    s4g_palette = lc_info["s4g_palette"]

    remapper = {}
    dict_palette = {}

    for x in range(len(values)):
        remapper[values[x]] = s4g_values_conversion[x]

    for x in range(len(s4g_values)):
        dict_palette[s4g_values[x]] = ImageColor.getrgb(f"#{s4g_palette[x]}")


# Generating indices
def generate_indices(table):
    for index_dict in INDICES:
        name = index_dict["name"]
        band1 = index_dict["band1"]
        band2 = index_dict["band2"]
        table[name] = (
            ((table[band1] / 1e4) - (table[band2] / 1e4))
            / ((table[band1] / 1e4) + (table[band2] / 1e4))
            * 1e4
        )


def model_lc(
    ms, dem, lc, points: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = points.copy()
    coords = [coord for coord in zip(points.geometry.x, points.geometry.y)]

    with TemporaryDirectory(delete=False) as folder:
        stack = f"{folder}/stack.vrt"

        logger.info("Make stack")
        check_call(
            f"""gdal raster pipeline \
                ! stack --resolution=average {ms} {dem} {lc} \
                ! set-type --ot=Int16 \
                ! write -f VRT {stack} \
            """,
            shell=True,
        )

        logger.info("Extract sample")
        with rio.open(stack) as src:
            points[COLUMNS] = [data for data in src.sample(coords)]
            points = points[
                (points[LABEL_LC] != 0)  # Filter no land cover
                & (points[LABEL_LC] != 32767)  # Filter no land cover
                & (points["BLUE"] > 0)  # Filter no data
                & (points[LABEL_LC] != 20122)  # Filter transmigration area
            ]
        points = points.replace({f"{LABEL_LC}": remapper})

        logger.info("Generate indices")
        generate_indices(points)

        # filter using indices
        points = points[
            ~(
                (  # Filter settlement and built up that have high NDMI and MNDWI2
                    (points["LC"] >= 60)
                    & (points["LC"] < 80)
                    & (points["NDMI"] > 0)
                    & (points["MNDWI2"] > 0)
                )
                | (  # Filter water that has low MNDWI2 and not flat
                    (points["LC"] == 80) & (points["MNDWI2"] <= 0)
                )
                | (  # Filter vegetation that has low NDMI and MNDWI bigger than 0
                    (points["LC"] <= 55) & (points["NDMI"] <= 0) & (points["MNDWI"] > 0)
                )
                | (  # Filter wet shrub vegetation that is low MNDWI2
                    (points["LC"] == 55) & (points["MNDWI2"] <= 0)
                )
            )
        ]

        lc_classes = points[LABEL_LC].unique()

        train_df = []
        test_df = []

        logger.info("Filter sample")
        for value in lc_classes:
            sample = points[points[LABEL_LC] == value]

            logger.info(f"Sample {value}: {len(sample)}")

            multipliers = 5
            means = sample[PREDICTORS].mean()
            std = sample[PREDICTORS].std()

            # Filter outlier values
            for var in PREDICTORS:
                sample = sample[
                    (sample[var] <= (means[var] + (multipliers * std[var])))
                    & (sample[var] >= (means[var] - (multipliers * std[var])))
                ]

            # only use sample which have sample more than 100
            if len(sample) > 100:
                if len(sample) > 5000:
                    sample = sample.sample(5000, random_state=RANDOM_STATE)

                # split into train and test per class values
                train, test = train_test_split(
                    sample, test_size=TEST_RATIO, random_state=RANDOM_STATE
                )
                train_df.append(train)
                test_df.append(test)

        del points

    train_df = pd.concat(train_df)
    test_df = pd.concat(test_df)

    return train_df, test_df


def load_image_to_array(
    image_path: str,
    array: np.ndarray,
    indices: tuple[int, int],
):
    logger.info(f"Load {image_path}")
    with rio.open(image_path) as src:
        window = windows.from_bounds(*BBOX, transform=src.transform)
        array[indices[0] : indices[1]] = src.read(
            out_shape=(HEIGHT, WIDTH), out_dtype="int16", window=window, boundless=True
        )


def load_image_and_apply_model(ms, dem, model, year, le: LabelEncoder):
    all_images = np.zeros(
        shape=(len(BANDS), HEIGHT, WIDTH),
        dtype="int16",
    )

    logger.info(f"Load images {year}")
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        jobs = [
            executor.submit(
                load_image_to_array,
                image_path=ms,
                array=all_images,
                indices=(0, len(MS_BANDS)),
            ),
            executor.submit(
                load_image_to_array,
                image_path=dem,
                array=all_images,
                indices=(
                    len(MS_BANDS),
                    len(MS_BANDS) + len(DEM_BANDS),
                ),
            ),
        ]

        for job in jobs:
            try:
                job.result()
            except Exception as e:
                logger.info(f"Error {year}: {e}")

    # Table image
    table_images = all_images.transpose(1, 2, 0)

    del all_images

    table_images = pd.DataFrame(
        table_images.reshape(-1, table_images.shape[2]),
        columns=BANDS,
    )

    # Apply indices to valid table
    logger.info(f"Generate indices {year}")
    generate_indices(table_images)

    # Valid mask
    valid_mask = table_images["BLUE"] > 0

    # Valid table
    valid_table = table_images[valid_mask]

    # Apply model
    logger.info(f"Run model classification {year}")
    classified_class = model.predict(valid_table[PREDICTORS])
    classified_class = le.inverse_transform(classified_class)

    del valid_table

    logger.info(f"Save result {year}")

    table_images.loc[valid_mask, LABEL_LC] = classified_class
    table_images.loc[~valid_mask, LABEL_LC] = 0

    del valid_mask

    image = table_images[LABEL_LC].to_numpy().reshape(HEIGHT, WIDTH)

    del table_images

    # load admin
    with rio.open(ADM) as src_adm:
        window = windows.from_bounds(*BBOX, transform=src_adm.transform)
        adm_image = (
            src_adm.read(1, out_shape=(HEIGHT, WIDTH), window=window, boundless=True)
            == 1
        )

    # mask with admin data
    image = image.copy()
    image[~adm_image] = 0

    # create image model
    with NamedTemporaryFile(suffix=".tif") as tmp:
        with rio.open(
            tmp.name,
            "w",
            "COG",
            count=1,
            width=WIDTH,
            height=HEIGHT,
            crs="EPSG:4326",
            transform=TARGET_TRANSFORM,
            nodata=0,
            dtype="uint8",
            compress="zstd",
            resampling="mode",
            **{"STATISTICS": "YES"},
        ) as src:
            src.write(image, 1)
            src.set_band_description(1, LABEL_LC)
            src.write_colormap(1, dict_palette)
            src.update_tags(1, CATEGORY_NAMES=str(s4g_labels))

        check_call(
            f"""gdal pipeline \
                ! read {tmp.name} \
                ! neighbors --method=mode --kernel=equal --size=3 \
                ! polygonize --attribute-name=LC \
                ! materialize \
                ! rasterize -a LC --init=0 --nodata=0 --resolution={RESOLUTION / 111_000},{RESOLUTION / 111_000} --ot=Byte \
                ! write -f COG --overwrite --co="COMPRESS=ZSTD" --co="STATISTICS=YES" --co="RESAMPLING=MODE" --co="OVERVIEWS=IGNORE_EXISTING" --co="OVERVIEW_RESAMPLING=MODE" {tmp.name} \
            """,
            shell=True,
        )

        with rio.open(tmp.name, "r+", **{"IGNORE_COG_LAYOUT_BREAK": "YES"}) as o:
            o.write_colormap(1, dict_palette)
            o.descriptions = tuple(["LC"])

        copyfile(
            tmp.name,
            f"{OUTPUT_PREFIX}/prediction_lc_klhk_s4g_v1/{REGION_NAME}_lc_{year}.tif",
        )


def main():
    grids_df = gpd.read_file(ROI)
    points = grids_df.sample_points(SAMPLE_SIZE).explode().reset_index()

    # 1. Initialize the LabelEncoder
    le = LabelEncoder()

    with ThreadPoolExecutor(4) as executor:
        jobs = [
            executor.submit(model_lc, EOS[0]["ms"], DEM, LCS[0]["lc"], points),  # 2000
            executor.submit(model_lc, EOS[2]["ms"], DEM, LCS[1]["lc"], points),  # 2011
            executor.submit(model_lc, EOS[3]["ms"], DEM, LCS[2]["lc"], points),  # 2015
            executor.submit(model_lc, EOS[4]["ms"], DEM, LCS[4]["lc"], points),  # 2020
        ]
        results = [job.result() for job in jobs]

    train_df = pd.concat([result[0] for result in results])
    test_df = pd.concat([result[1] for result in results])
    train_encoded = le.fit_transform(train_df[LABEL_LC])

    logger.info("Train model")
    # model = RandomForestClassifier(100, class_weight="balanced")
    model = XGBRFClassifier(
        n_estimators=100,
        max_depth=16,
        max_leaves=0,
        sample_weight=compute_sample_weight("balanced", train_encoded),
    )
    model.fit(train_df[PREDICTORS], train_encoded)

    test_apply = model.predict(test_df[PREDICTORS])
    test_apply = le.inverse_transform(test_apply)

    cm = confusion_matrix(test_df[LABEL_LC], test_apply)
    report = classification_report(test_df[LABEL_LC], test_apply)
    accuracy = accuracy_score(test_df[LABEL_LC], test_apply)
    kappa = cohen_kappa_score(test_df[LABEL_LC], test_apply)
    # f1 = f1_score(test_df[LABEL_LC], test_apply)
    # recall = recall_score(test_df[LABEL_LC], test_apply)
    # precision = precision_score(test_df[LABEL_LC], test_apply)

    logger.info(cm)
    logger.info(report)

    # logger.info(f"F1={f1}")
    # logger.info(f"Recall={recall}")
    # logger.info(f"Precision={precision}")

    metrics_path = f"{OUTPUT_PREFIX}/model_rf_result_v1.json"
    metrics = {
        "SAMPLE": {"TRAIN": len(train_df), "TEST": len(test_df)},
        "MODEL": {
            "ACCURACY": accuracy,
            "KAPPA": kappa,
            # "F1": f1,
            # "RECALL": recall,
            # "PRECISION": precision,
        },
    }
    with open(metrics_path, "w") as file:
        json.dump(metrics, file)

    for dict_data in EOS:
        ms = dict_data["ms"]
        year = dict_data["year"]
        load_image_and_apply_model(ms, DEM, model, year, le)


if __name__ == "__main__":
    main()
