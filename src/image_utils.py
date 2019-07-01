import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def load_image_array(
    image_path: str, target_size: tuple = None, color_mode: str = "rgb"
) -> np.ndarray:
    load_img = tf.keras.preprocessing.image.load_img
    img_to_array = tf.keras.preprocessing.image.img_to_array
    image_array = img_to_array(load_img(image_path, target_size=target_size))
    return np.expand_dims(image_array, axis=0)


def save_image_array(image_path: str, array: np.ndarray):
    pillow_image = array_to_img(array)
    pillow_image.save(image_path)
