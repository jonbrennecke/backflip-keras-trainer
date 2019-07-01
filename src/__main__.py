import os
import math
import tensorflow as tf
import numpy as np

from .constants import DATA_DIR_PATH, IMAGE_DIMENSIONS
from .dataset import gen_training_files
from .model import Model

load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array

image_dimensions = IMAGE_DIMENSIONS["original"]


def load_image_array(image_path: str, target_size: tuple = None) -> np.ndarray:
    load_img = tf.keras.preprocessing.image.load_img
    img_to_array = tf.keras.preprocessing.image.img_to_array
    image_array = img_to_array(load_img(image_path, target_size=target_size))
    return np.expand_dims(image_array, axis=0)


def train():
    model = Model()
    model.compile()
    # print(model.summary())

    for dataset_files in gen_training_files(DATA_DIR_PATH):
        dataset_path = os.path.join(DATA_DIR_PATH, dataset_files["dirname"])
        images = dataset_files["images"]
        depth_image_path = os.path.join(dataset_path, images["depth"])
        color_image_path = os.path.join(dataset_path, images["color"])
        segmentation_image_path = os.path.join(dataset_path, images["segmentation"])

        # resize both color and depth images to expected size
        color_image_width, color_image_height = image_dimensions["color_image"][0:2]
        color_image_array = load_image_array(
            color_image_path, target_size=(color_image_height, color_image_width)
        )
        depth_image_array = load_image_array(
            depth_image_path, target_size=(color_image_height, color_image_width)
        )

        segmentation_image_width, segmentation_image_height = image_dimensions[
            "segmentation_image"
        ][0:2]
        
        
        _, segmentation_image_height, segmentation_image_width, _ = model.model.output_shape
        segmentation_image_array = load_image_array(
            segmentation_image_path,
            target_size=(segmentation_image_height, segmentation_image_width),
        )
        model.train(color_image_array, depth_image_array, segmentation_image_array)


def main():
    train()


if __name__ == "__main__":
    main()
