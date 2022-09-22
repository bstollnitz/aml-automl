"""Generates images for inference."""

import logging
import os
from pathlib import Path
from PIL import Image
import numpy as np

from torchvision import datasets

DATA_DIR = "aml_batch_endpoint/test_data/data"
IMAGES_DIR = "aml_batch_endpoint/test_data/images"
TEST_DATA_DIR = "aml_batch_endpoint/test_data"


def generate_images(num_images: int) -> None:
    """
    Generates images for inference.
    """
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=True)

    images_dir = Path(IMAGES_DIR)
    if images_dir.exists():
        for f in Path(IMAGES_DIR).iterdir():
            f.unlink()
    else:
        os.makedirs(IMAGES_DIR)

    for i, (image, _) in enumerate(test_data):
        if i == num_images:
            break
        image.save(f"{IMAGES_DIR}/image_{i+1:0>3}.png")


def generate_csv_from_images() -> None:
    """
    Generates CSV file from the images.
    """
    delimiter = ","
    fmt = "%.6f"
    image_paths = [f for f in Path(IMAGES_DIR).iterdir() if Path.is_file(f)]
    image_paths.sort()

    X = None
    for (i, image_path) in enumerate(image_paths):
        with Image.open(image_path) as image:
            if X is None:
                size = image.height * image.width
                X = np.empty((len(image_paths), size))
            x = np.asarray(image).reshape((-1))
            X[i, :] = x

    header = delimiter.join([f"col_{i}" for i in range(X.shape[1])])
    np.savetxt(fname=Path(TEST_DATA_DIR, "images.csv"),
               X=X,
               delimiter=delimiter,
               fmt=fmt,
               header=header,
               comments="")


def main():
    logging.basicConfig(level=logging.INFO)

    generate_images(20)
    generate_csv_from_images()


if __name__ == "__main__":
    main()
