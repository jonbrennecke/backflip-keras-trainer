import tensorflow as tf
import numpy as np
from PIL import Image


def load_image_array(
    image_path: str, size: tuple = None, color_mode: str = "rgb"
) -> np.ndarray:
    width, height = size
    img = Image.open(image_path).convert('L')
    resized_img = img.resize((height, width), Image.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
    return np.expand_dims(img_array / 255, axis=0)


def save_image_array(image_path: str, array: np.ndarray):
    pillow_image = tf.keras.preprocessing.image.array_to_img(array)
    pillow_image.save(image_path)
