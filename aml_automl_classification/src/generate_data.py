"""Generates data for training."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

TRAIN_DATA_DIR = "aml_automl_classification/cloud/automl_train_data"
TEST_DATA_DIR = "aml_automl_classification/cloud/automl_test_data"
BATCH_SIZE = 64


def generate_csv(train: bool) -> None:
    """
    Generates CSV file containing training or test data.
    Each row in the CSV file contains the 784 pixels of an image, followed by
    the corresponding label.
    """
    delimiter = ","
    fmt = "%.6f"

    data_dir = TRAIN_DATA_DIR if train else TEST_DATA_DIR
    data = datasets.FashionMNIST(data_dir,
                                 train=train,
                                 download=True,
                                 transform=ToTensor())
    loader: DataLoader[torch.Tensor] = DataLoader(data,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)

    image_nelements = data[0][0].nelement()
    csv_width = image_nelements + 1
    csv_height = len(data)

    csv_matrix = np.empty((csv_height, csv_width))
    for (i, (x, y)) in enumerate(tqdm(loader)):
        x = x.view(x.size(0), -1)
        csv_matrix[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, 0:-1] = x
        csv_matrix[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE, -1] = y

    header_list = [f"col_{i}" for i in range(csv_width - 1)]
    header_list.append("label")
    csv_header = delimiter.join(header_list)
    np.savetxt(fname=Path(data_dir, "data.csv"),
               X=csv_matrix,
               delimiter=delimiter,
               fmt=fmt,
               header=csv_header,
               comments="")

    mltable = Path(data_dir, "MLTable")
    mltable_contents = f"""
paths:
    - file: ./data.csv
transformations:
    - read_delimited:
        delimiter: '{delimiter}'
        encoding: 'ascii'"""
    with open(mltable, "w", encoding="utf-8") as f:
        f.write(mltable_contents)


def main():
    logging.basicConfig(level=logging.INFO)

    generate_csv(train=True)
    generate_csv(train=False)


if __name__ == "__main__":
    main()
