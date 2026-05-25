import json
import logging
from datetime import datetime
from os import listdir

import numpy as np
import rasterio as rio
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPool2D,
)
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU
from keras.models import load_model
from keras.optimizers import Adam
from rasterio.enums import Resampling
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from job.utils import MAX_WORKERS, logger

# Silence the rasterio logger
logging.getLogger("rasterio").setLevel(logging.ERROR)

# If the warnings are coming from GDAL directly, silence that too
logging.getLogger("rasterio._gdal").setLevel(logging.ERROR)

MAX_WORKERS = MAX_WORKERS

PREDICTORS_PREFIX = "/usr/src/app/output/tile_images/image"
LABELS_PREFIX = "/usr/src/app/output/tile_images/label"
OUTPUT_PATH = "/usr/src/app/output"

# Sampling parameter
RANDOM_STATE = 1
SAMPLE_COUNT = 6472
TEST_RATIO = 0.25
IMAGE_SIZE = 128
BANDS_COUNT = 3

# modelling parameter
NEURONS = 32
KERNEL = 3
PADDING = "same"
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
VALIDATION_SPLIT = 0.5
MAX_POOL = 2
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_NAME = f"unet_chm_v1_{IMAGE_SIZE}x{IMAGE_SIZE}_sampleCount{SAMPLE_COUNT}_{str(round(datetime.now().timestamp()))}"
AUTOTUNE = tf.data.AUTOTUNE

# Modelling param json
model_param_json = dict(
    SAMPLING_PARAM=dict(
        RANDOM_STATE=RANDOM_STATE,
        SAMPLE_COUNT=SAMPLE_COUNT,
        TEST_RATIO=TEST_RATIO,
        IMAGE_SIZE=IMAGE_SIZE,
        BANDS_COUNT=BANDS_COUNT,
    ),
    MODELLING_PARAM=dict(
        NEURONS=NEURONS,
        KERNEL=KERNEL,
        PADDING=PADDING,
        DROPOUT=DROPOUT,
        VALIDATION_SPLIT=VALIDATION_SPLIT,
        MAX_POOL=MAX_POOL,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS,
        MODEL_NAME=MODEL_NAME,
        LEARNING_RATE=LEARNING_RATE,
        WEIGHT_DECAY=WEIGHT_DECAY,
    ),
)


def get_id(path):
    return path.split("/")[-1].split("_")[0]


def prepare_sample():
    prefix = LABELS_PREFIX
    paths = listdir(prefix)

    # get random image
    logger.info(f"Get random {SAMPLE_COUNT} sample")
    # paths = pd.Series(paths).sample(SAMPLE_COUNT, random_state=RANDOM_STATE).to_list()
    ids = [get_id(path) for path in paths]

    # split id to train and test
    logger.info("Split sample train and test")
    train, test = train_test_split(ids, test_size=TEST_RATIO)
    test, validation = train_test_split(test, test_size=VALIDATION_SPLIT)

    # Dictionary of data path
    data_dict = dict(
        train=dict(
            images=[f"{PREDICTORS_PREFIX}/{id}_IMAGE.tif" for id in train],
            labels=[f"{prefix}/{id}_LABEL.tif" for id in train],
        ),
        test=dict(
            images=[f"{PREDICTORS_PREFIX}/{id}_IMAGE.tif" for id in test],
            labels=[f"{prefix}/{id}_LABEL.tif" for id in test],
        ),
        validation=dict(
            images=[f"{PREDICTORS_PREFIX}/{id}_IMAGE.tif" for id in validation],
            labels=[f"{prefix}/{id}_LABEL.tif" for id in validation],
        ),
    )

    return data_dict


def load_image_tf_dataset(image_path, label_path):
    image_path = image_path.numpy().decode("utf-8")
    label_path = label_path.numpy().decode("utf-8")

    with rio.open(image_path, **{"IGNORE_COG_LAYOUT_BREAK": True}) as src:
        image = (
            src.read(
                out_shape=(IMAGE_SIZE, IMAGE_SIZE),
                out_dtype="float32",
                resampling=Resampling.nearest,
            ).transpose(1, 2, 0)
            / 255
        )

    with rio.open(label_path, **{"IGNORE_COG_LAYOUT_BREAK": True}) as src:
        label = src.read(
            [1],
            out_shape=(IMAGE_SIZE, IMAGE_SIZE),
            resampling=Resampling.nearest,
        ).transpose(1, 2, 0)

    return image, label


# --- 3. The Wrapper (Crucial Step) ---
def tf_dataset_wrapper(image_path, label_path):
    img, lbl = tf.py_function(
        func=load_image_tf_dataset,
        inp=[image_path, label_path],
        Tout=[tf.float32, tf.uint8],  # Define output types
    )

    # Explicitly set shapes (py_function loses shape info)
    img.set_shape([IMAGE_SIZE, IMAGE_SIZE, BANDS_COUNT])
    lbl.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])

    return img, lbl


def dual_conv2_block(neuron, input):
    conv1 = Conv2D(
        neuron,
        kernel_size=KERNEL,
        padding=PADDING,
        activation="relu",
    )(input)
    conv2 = Conv2D(
        neuron,
        kernel_size=KERNEL,
        padding=PADDING,
        activation="relu",
    )(conv1)
    return conv2


def encode_block(neuron, input):
    dual_conv = dual_conv2_block(neuron, input)
    dropout = Dropout(DROPOUT)(dual_conv)
    max_pool = MaxPool2D(MAX_POOL)(dropout)
    return max_pool, dropout


def decode_block(neuron, input, pair):
    transpose = Conv2DTranspose(
        neuron, kernel_size=KERNEL, strides=2, padding=PADDING, activation="relu"
    )(input)
    concat = Concatenate()([pair, transpose])
    dual_conv = dual_conv2_block(neuron, concat)
    dropout = Dropout(DROPOUT)(dual_conv)
    return dropout


def unet_model(train_dataset, validation_dataset):
    input = Input((None, None, BANDS_COUNT))

    encode_1, match_1 = encode_block(NEURONS * 1, input)
    encode_2, match_2 = encode_block(NEURONS * 2, encode_1)
    encode_3, match_3 = encode_block(NEURONS * 4, encode_2)
    encode_4, match_4 = encode_block(NEURONS * 8, encode_3)
    transition = dual_conv2_block(NEURONS * 16, encode_4)
    decode_4 = decode_block(NEURONS * 8, transition, match_4)
    decode_3 = decode_block(NEURONS * 4, decode_4, match_3)
    decode_2 = decode_block(NEURONS * 2, decode_3, match_2)
    decode_1 = decode_block(NEURONS * 1, decode_2, match_1)
    output = Conv2D(
        1,
        kernel_size=KERNEL,
        padding=PADDING,
        activation="sigmoid",
    )(decode_1)

    model = Model(input, output)
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryIoU(),
        ],
    )

    callbacks = [
        EarlyStopping(
            patience=5,
            monitor="binary_io_u",
            mode="max",
        )
    ]

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_dataset,
        class_weight={0: 1, 1: 5},
        shuffle=True,
    )

    return model


def main():
    # Get labels list
    results = prepare_sample()

    # Get labels and images for train and test path only
    train_labels_path = results["train"]["labels"]
    train_images_path = results["train"]["images"]
    test_labels_path = results["test"]["labels"]
    test_images_path = results["test"]["images"]
    validation_labels_path = results["validation"]["labels"]
    validation_images_path = results["validation"]["images"]

    # Datasets tf
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_images_path, train_labels_path))
        .map(tf_dataset_wrapper, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    validation_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (validation_images_path, validation_labels_path)
        )
        .map(tf_dataset_wrapper, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_images_path, test_labels_path))
        .map(tf_dataset_wrapper, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    # Train model
    model = unet_model(train_dataset, validation_dataset)

    # Save the model
    logger.info("Save model")
    saved_model_path = f"{OUTPUT_PATH}/{MODEL_NAME}.keras"
    model.save(saved_model_path)

    model = load_model(saved_model_path)

    # Predict test
    logger.info("Predict test data")
    test_images_predicted = model.predict(test_dataset, batch_size=BATCH_SIZE)
    test_images_predicted = np.round(test_images_predicted).flatten()

    # Get flatten version of the label
    label_true = [label for (_, label) in test_dataset.as_numpy_iterator()]
    label_true = np.concat(label_true, 0).flatten()

    # Compares test labels and prediction
    logger.info("Assess model")
    accuracy = accuracy_score(label_true, test_images_predicted)
    kappa = cohen_kappa_score(label_true, test_images_predicted)
    report = classification_report(label_true, test_images_predicted)
    recall = recall_score(label_true, test_images_predicted)
    precision = precision_score(label_true, test_images_predicted)
    f1 = f1_score(label_true, test_images_predicted)
    auc = roc_auc_score(label_true, test_images_predicted)
    iou = jaccard_score(label_true, test_images_predicted)
    logger.info(report)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Kappa: {kappa}")
    logger.info(f"Recall: {recall}")
    logger.info(f"Precision: {precision}")
    logger.info(f"F1: {f1}")
    logger.info(f"AUC: {auc}")
    logger.info(f"IOU: {iou}")

    # Save parameter of the model
    model_param_json["METRICS"] = {
        "ACCURACY": accuracy,
        "KAPPA": kappa,
        "RECALL": recall,
        "PRECISION": precision,
        "F1": f1,
        "AUC": auc,
        "IOU": iou,
    }
    model_param_path = f"{OUTPUT_PATH}/{MODEL_NAME}.json"
    with open(model_param_path, "w") as file:
        json.dump(model_param_json, file)


if __name__ == "__main__":
    main()
