import os


def list_subdirectory_names(dir_path: str) -> list:
    if not os.path.isdir(dir_path):
        raise Exception(f"Provided data directory is not a directory: {dir_path}")
    walk = os.walk(dir_path)
    (_, dirnames, _) = next(walk)
    return dirnames


def list_filenames(dataset_path: str) -> list:
    if not os.path.isdir(dataset_path):
        raise Exception(
            f"Provided dataset directory is not a directory: {dataset_path}"
        )
    walk = os.walk(dataset_path)
    (_, _, filenames) = next(walk)
    return filenames


def list_dataset_training_image_filenames(dataset_path: str) -> list:
    filenames = list_filenames(dataset_path)
    depth_filename = None
    segmentation_filename = None
    color_filename = None
    for filename in filenames:
        (name, ext) = os.path.splitext(filename)
        lower_name = name.lower()
        if lower_name == "depth":
            depth_filename = filename
        elif lower_name == "color":
            color_filename = filename
        elif lower_name == "segmentation":
            segmentation_filename = filename
        else:
            raise Exception(f"unexpected file: {filename} in {dataset_path}")
    if not all([depth_filename, segmentation_filename, color_filename]):
        raise Exception(f"missing expected files in {dataset_path}")
    return {
        "depth": depth_filename,
        "color": color_filename,
        "segmentation": segmentation_filename,
    }


def list_dataset_debug_image_filenames(dataset_path: str) -> list:
    dataset_images_path = os.path.join(dataset_path)
    filenames = list_filenames(dataset_images_path)


def gen_training_files(root_dir_path, images_dir_name: str = "training_images"):
    dir_path = os.path.join(root_dir_path, images_dir_name)
    subdirs = list_subdirectory_names(dir_path)
    for subdir in subdirs:
        subdir_path = os.path.join(dir_path, subdir)
        images = list_dataset_training_image_filenames(subdir_path)
        yield {"path": subdir_path, "images": images}


def gen_debug_files(root_dir_path, images_dir_name: str = "debug_images"):
    dir_path = os.path.join(root_dir_path, images_dir_name)
    subdirs = list_subdirectory_names(dir_path)
    for subdir in subdirs:
        subdir_path = os.path.join(dir_path, subdir)
        images = list_dataset_training_image_filenames(subdir_path)
        yield {"path": subdir_path, "images": images}
