import tensorflow as tf
import numpy as np


def load_image_array(
    image_path: str, target_size: tuple = None, color_mode: str = "rgb"
) -> np.ndarray:
    image_array = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=target_size,
            color_mode=color_mode,
            interpolation="lanczos",
        )
    )
    return np.expand_dims(image_array, axis=0)


def save_image_array(image_path: str, array: np.ndarray):
    pillow_image = tf.keras.preprocessing.image.array_to_img(array)
    pillow_image.save(image_path)
