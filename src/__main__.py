import os
import math
import secrets
import tensorflow as tf
import numpy as np

from .constants import DATA_DIR_PATH, IMAGE_DIMENSIONS, H5_MODEL_PATH, COREML_MODEL_PATH
from .dataset import gen_training_files, gen_debug_files
from .model import Model
from .image_utils import load_image_array, save_image_array


image_dimensions = IMAGE_DIMENSIONS["original"]


def run_training(model: Model):
    for dataset_files in gen_training_files(DATA_DIR_PATH):
        dataset_path = os.path.join(DATA_DIR_PATH, dataset_files["path"])
        images = dataset_files["images"]
        color_image_path = os.path.join(dataset_path, images["color"])
        depth_image_path = os.path.join(dataset_path, images["depth"])
        segmentation_image_path = os.path.join(dataset_path, images["segmentation"])

        # resize both color and depth images to expected size
        color_image_width, color_image_height = image_dimensions["color_image"][0:2]
        color_image_array = load_image_array(
            color_image_path, target_size=(color_image_height, color_image_width)
        )

        depth_image_array = load_image_array(
            depth_image_path,
            target_size=(color_image_height, color_image_width),
            color_mode="grayscale",
        )

        _, segmentation_image_height, segmentation_image_width, _ = (
            model.model.output_shape
        )

        segmentation_image_array = load_image_array(
            segmentation_image_path,
            target_size=(segmentation_image_height, segmentation_image_width),
            color_mode="grayscale",
        )

        model.train(color_image_array, depth_image_array, segmentation_image_array)


def run_debug_prediction(model: Model):
    for dataset_files in gen_debug_files(DATA_DIR_PATH):
        dataset_path = os.path.join(DATA_DIR_PATH, dataset_files["path"])
        images = dataset_files["images"]
        color_image_path = os.path.join(dataset_path, images["color"])
        depth_image_path = os.path.join(dataset_path, images["depth"])

        color_image_width, color_image_height = image_dimensions["color_image"][0:2]
        color_image_array = load_image_array(
            color_image_path, target_size=(color_image_height, color_image_width)
        )

        depth_image_array = load_image_array(
            depth_image_path,
            target_size=(color_image_height, color_image_width),
            color_mode="grayscale",
        )

        prediction_image_array = model.predict(color_image_array, depth_image_array)

        reshaped = np.reshape(
            prediction_image_array,
            (prediction_image_array.shape[1], prediction_image_array.shape[2], 1),
        )
        filename = f"/Users/jon/Downloads/predict-{secrets.token_hex(10)}.jpg"
        save_image_array(filename, reshaped)
        print(f"Saved image to: {filename}")


def main():
    model = Model()
    model.compile()
    model.print_summary()
    run_training(model)
    run_debug_prediction(model)

    model.save_h5(H5_MODEL_PATH)
    print(f"Saved h5 model to: {H5_MODEL_PATH}")

    model.save_coreml(COREML_MODEL_PATH)
    print(f"Saved coreml model to: {COREML_MODEL_PATH}")


if __name__ == "__main__":
    main()
