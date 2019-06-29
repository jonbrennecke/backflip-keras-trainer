import os

from .constants import DATA_DIR_PATH
from .dataset import gen_training_files
from .model import Model


def train():
    model = Model()
    # for dataset_files in gen_training_files(DATA_DIR_PATH):
    #     dataset_path = os.path.join(DATA_DIR_PATH, dataset_files["dirname"])


def main():
    train()


if __name__ == "__main__":
    main()
