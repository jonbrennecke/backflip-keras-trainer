import os

DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH")

IMAGE_DIMENSIONS = dict(
    original=dict(
        color_image=(2320, 3088, 3),
        depth_image=(480, 640, 3),
        segmentation_image=(1160, 1544, 3),
    ),
    model=dict(
        color_image=(3088, 2320, 3),
        depth_image=(3088, 2320, 3),
        segmentation_image=(1544, 1160, 3),
    ),
)
