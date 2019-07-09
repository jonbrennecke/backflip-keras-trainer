import os

DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH")

scale = 0.25

IMAGE_DIMENSIONS = dict(
    original=dict(
        # color_image=(1024, 1920, 1),
        # depth_image=(1024, 1920, 1),
        color_image=(int(1024 * scale) , int(1920 * scale), 1),
        depth_image=(int(1024 * scale), int(1920 * scale), 1),
        segmentation_image=(1160, 1544, 1),
    ),
    model=dict(
        # color_image=(1920, 1024, 1),
        # depth_image=(1920, 1024, 1),
        # segmentation_image=(1544, 1160, 1),
        color_image=(int(1920 * scale) , int(1024 * scale), 1),
        depth_image=(int(1920 * scale), int(1024 * scale), 1)
    ),
)

H5_MODEL_PATH = os.environ.get("H5_MODEL_PATH")
COREML_MODEL_PATH = os.environ.get("COREML_MODEL_PATH")
