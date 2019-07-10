import os

DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH")
H5_MODEL_PATH = os.environ.get("H5_MODEL_PATH")
COREML_MODEL_PATH = os.environ.get("COREML_MODEL_PATH")
NUMBER_OF_STEPS_PER_EPOCH = int(os.environ.get("NUMBER_OF_STEPS_PER_EPOCH"))
NUMBER_OF_EPOCHS = int(os.environ.get("NUMBER_OF_EPOCHS"))

scale = 0.25
ratio = 1920 / 1080  # 16/9
base = 16  # width and height must be divisible by 'base'


def nearest_multiple(x):
    return base * round(x / base)


width = nearest_multiple(int(1024 * scale))
height = nearest_multiple(int(1024 * 16 / 9 * scale))

IMAGE_DIMENSIONS = dict(
    original=dict(color_image=(width, height, 1), depth_image=(width, height, 1)),
    model=dict(color_image=(height, width, 1), depth_image=(height, width, 1)),
)
