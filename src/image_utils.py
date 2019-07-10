import tensorflow as tf
import numpy as np


def load_image_array(
    image_path: str, size: tuple = None, color_mode: str = "rgb"
) -> np.ndarray:
    width, height = size
    offset = 10
    target_size = [width + offset, height + offset]
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        color_mode=color_mode,
        interpolation="lanczos",
        target_size=target_size
    )
    cropped_img = img.crop((offset, offset, height, width))
    resized_img = cropped_img.resize((height, width))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
    return np.expand_dims(img_array / 255, axis=0)


def save_image_array(image_path: str, array: np.ndarray):
    pillow_image = tf.keras.preprocessing.image.array_to_img(array)
    pillow_image.save(image_path)
