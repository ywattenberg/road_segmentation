import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import segmentation_models_pytorch as smp
from Dataset.dataset import ETHDataset
from torch.utils.data import DataLoader
import click
import subprocess
import sys

MODEL_PATHS = [
    "model_weights_UnetPlusPlus-resnet34-imagenet-clDice.pth",
    "model_weights_UnetPlusPlus-resnet34-imagenet-clDice.pth",
    "model_weights_Linknet-resnet34-imagenet-clDice.pth",
    "model_weights_Linknet-resnet50-imagenet-clDice.pth",
]
TEST_SCORES = [0.1911, 0.183857, 0.188096, 0.189461]


def get_model(model_name, encoder_name, model_path, device=None):
    model_func = getattr(smp, model_name)

    model = model_func(
        encoder_name=encoder_name,
        # activation="sigmoid",
        in_channels=4,
        classes=1,
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if device is not None:
        model.to(device)
    return model


def get_masks(model, dataset, batch_size, device=None):
    reshape = transforms.CenterCrop(400)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    masks = None
    for x in loader:
        images = x.to(device) if device is not None else x
        with torch.no_grad():
            pred = model(images)
        if masks is None:
            masks = pred
        else:
            masks = torch.cat([masks, pred], dim=0)
    return reshape(masks)


def calc_weights_from_scores(scores):
    scores = np.array(scores)
    weights = scores / scores.sum()
    weights = np.ones(weights.shape) - weights
    return weights


@click.command()
@click.option("--model_dir", default="models", help="Directory with models")
@click.option("--weighted", "-w", is_flag=True, help="Use weighted average")
@click.option("--device", "-d", default=None, help="Device to use")
@click.option("--batch_size", "-b", default=8, help="Batch size")
@click.option("--output", "-o", default="submission.csv", help="Output file")
@click.option("--threshold", "-t", default=0.4, help="Threshold for mask")
@click.option(
    "--train", "-t", is_flag=True, help="Use training set instead of test set"
)
def main(model_dir, weighted, device, batch_size, output, threshold, train):
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_path = "data/ethz-cil-road-segmentation-2023"
    if not train:
        image_path = os.path.join(base_path, "test/images")
    else:
        image_path = os.path.join(base_path, "training/images")
    # mask_path = os.path.join(base_path, "training/groundtruth")
    dataset = ETHDataset(
        image_path, None, augment_images=False, normalize=False, submission=True
    )

    model_dir = "models"
    if weighted:
        weights = calc_weights_from_scores(TEST_SCORES)
    else:
        weights = np.ones(len(MODEL_PATHS)) / len(MODEL_PATHS)

    masks = []
    for path, weight in zip(MODEL_PATHS, weights):
        model = path.split("_")[-1]
        model, encoder = model.split("-")[0:2]
        print(f"Using model {model} with encoder {encoder} and weight {weight}")
        model = get_model(model, encoder, os.path.join(model_dir, path), device=device)
        masks.append(get_masks(model, dataset, batch_size=batch_size, device=device))

    masks = [mask.cpu().numpy() for mask in masks]
    masks = [mask * weight for mask, weight in zip(masks, weights)]
    masks = np.array(masks).sum(axis=0)
    masks = ((masks > threshold) * 255).astype(np.uint8)

    for mask, name in zip(masks, os.listdir(image_path)):
        mask = Image.fromarray(mask.squeeze())
        mask.save(os.path.join("submission", name))
    if not train:
        subprocess.run(
            [
                os.environ["PYTHONPATH"],
                "src/mask_to_submission.py",
                "--base_dir",
                "submission",
                "--submission_filename",
                output,
            ]
        )


if __name__ == "__main__":
    main()
