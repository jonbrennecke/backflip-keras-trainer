import os

DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH")

IMAGE_DIMENSIONS = dict(
    original=dict(
        color_image=(1080, 1920, 3),
        depth_image=(1080, 1920, 3),
        segmentation_image=(1160, 1544, 3),
    ),
    model=dict(
        color_image=(1920, 1080, 3),
        depth_image=(1920, 1080, 1),
        segmentation_image=(1544, 1160, 1),
    ),
)

H5_MODEL_PATH = os.environ.get("H5_MODEL_PATH")
COREML_MODEL_PATH = os.environ.get("COREML_MODEL_PATH")
