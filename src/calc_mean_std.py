import torch
import multiprocessing
import numpy as np
import os
import sys
from Dataset.dataset import GMapsDataset


def mean_square(image, array):
    image = image.numpy()
    array[0] += np.mean(image, axis=(1, 2))
    array[1] += np.sum(image**2, axis=(1, 2)) / (image.shape[1] * image.shape[2])


if __name__ == "__main__":
    base_path = "data/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    skeleton_path = os.path.join(base_path, "skel")
    dataset = GMapsDataset(
        image_path, mask_path, skel_path=skeleton_path, augment_images=False
    )

    arr = [np.zeros(4), np.zeros(4)]
    for image, _, _ in dataset:
        mean_square(image, arr)

    mean = arr[0] / len(dataset)
    std = np.sqrt(arr[1] / (len(dataset)) - mean**2)
    print(mean)
    print(std)
