import os
import math
import random
import secrets
import tensorflow as tf
import numpy as np

from .constants import DATA_DIR_PATH, IMAGE_DIMENSIONS, H5_MODEL_PATH, COREML_MODEL_PATH
from .dataset import gen_training_files, gen_debug_files
from .model import Model
from .image_utils import load_image_array, save_image_array


image_dimensions = IMAGE_DIMENSIONS["original"]


def run_training(model: Model):
    training_data_gen = gen_training_data(model)
    model.train_generator(make_infinite_random_generator(training_data_gen))


def make_infinite_random_generator(input_generator):
    data = list(input_generator)

    def infinite_random_generator():
        while True:
            choice = random.choice(data)
            yield flip_images_randomly(choice)

    return infinite_random_generator()


def random_boolean_choice() -> bool:
    return random.choice([True, False])


def flip_images_randomly(input_images: tuple):
    if random_boolean_choice():
        return tuple(map(lambda x: np.fliplr(x), input_images))
    return input_images


def gen_training_data(model: Model):
    for dataset_files in gen_training_files(DATA_DIR_PATH):
        dataset_path = os.path.join(DATA_DIR_PATH, dataset_files["path"])
        images = dataset_files["images"]
        color_image_path = os.path.join(dataset_path, images["color"])
        depth_image_path = os.path.join(dataset_path, images["depth"])
        segmentation_image_path = os.path.join(dataset_path, images["segmentation"])

        # resize both color and depth images to expected size
        color_image_width, color_image_height = image_dimensions["color_image"][0:2]
        color_image_array = load_image_array(
            color_image_path,
            target_size=(color_image_height, color_image_width),
            color_mode="grayscale",
        )

        depth_image_width, depth_image_height = image_dimensions["depth_image"][0:2]
        depth_image_array = load_image_array(
            depth_image_path,
            target_size=(depth_image_height, depth_image_width),
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

        yield (color_image_array, depth_image_array, segmentation_image_array)


def run_debug_prediction(model: Model):
    for dataset_files in gen_debug_files(DATA_DIR_PATH):
        dataset_path = os.path.join(DATA_DIR_PATH, dataset_files["path"])
        images = dataset_files["images"]
        color_image_path = os.path.join(dataset_path, images["color"])
        depth_image_path = os.path.join(dataset_path, images["depth"])

        color_image_width, color_image_height = image_dimensions["color_image"][0:2]
        color_image_array = load_image_array(
            color_image_path,
            target_size=(color_image_height, color_image_width),
            color_mode="grayscale",
        )

        depth_image_width, depth_image_height = image_dimensions["depth_image"][0:2]
        depth_image_array = load_image_array(
            depth_image_path,
            target_size=(depth_image_height, depth_image_width),
            color_mode="grayscale",
        )

        prediction_image_array = (
            model.predict(color_image_array, depth_image_array) * 255
        )

        token = secrets.token_hex(10)
        reshaped_prediction_image_array = np.reshape(
            prediction_image_array,
            (prediction_image_array.shape[1], prediction_image_array.shape[2], 1),
        )
        filename_prediction = f"/Users/jon/Downloads/{token}-prediction.jpg"
        save_image_array(filename_prediction, reshaped_prediction_image_array)
        print(f"Saved prediction image to: {filename_prediction}")

        reshaped_original_image_array = np.reshape(
            color_image_array,
            (color_image_array.shape[1], color_image_array.shape[2], color_image_array.shape[3]),
        )
        filename_original = f"/Users/jon/Downloads/{token}-original.jpg"
        save_image_array(filename_original, reshaped_original_image_array)
        print(f"Saved original image to: {filename_original}")


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
