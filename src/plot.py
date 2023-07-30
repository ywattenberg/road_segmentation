import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_model_comp(csv_dir="model-comparison"):
    fig = plt.figure(figsize=(9, 8))
    plt.style.use("ggplot")
    for file in sorted(os.listdir(csv_dir)):
        # if not file.endswith(".csv") or not file.startswith("Unet"):
        #     continue

        df = pd.read_csv(os.path.join(csv_dir, file))

        df = df[df["epoch"] < 10]
        plt.plot(df["epoch"], df["DiceLoss"], label=file.split("-")[0])
    plt.legend()
    plt.tight_layout(pad=3)
    plt.title("Different models with resnet34 encoder")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    fig.savefig(f"figures/model-comparison-10.png", dpi=1000)


def plot_encoder_comp_unetpp(csv_dir="encoder-comparison"):
    fig = plt.figure(figsize=(9, 8))
    plt.style.use("ggplot")
    for file in sorted(os.listdir(csv_dir)):
        if not file.endswith(".csv") or not file.startswith("Unet"):
            continue

        df = pd.read_csv(os.path.join(csv_dir, file))
        df = df[df["epoch"] < 30]
        plt.plot(df["epoch"], df["DiceLoss"], label=file.split("-")[1])
    # plt.ylim(0.1, 0.4)
    plt.legend()
    plt.tight_layout(pad=3)
    plt.title("UnetPlutPlus model with different encoders")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    fig.savefig(f"figures/encoder-comparison-unetpp-30.png", dpi=1000)


def plot_model_encoder_comp(csv_dir="encoder-comparison"):
    fig = plt.figure(figsize=(9, 8))
    plt.style.use("ggplot")
    for file in sorted(os.listdir(csv_dir)):
        if not file.endswith(".csv") or (
            file.startswith("Unet")
            and not file.endswith("efficientnet-b5-imagenet-clDice.csv")
        ):
            continue

        df = pd.read_csv(os.path.join(csv_dir, file))
        df = df[df["epoch"] < 30]

        if file.split("-")[1] != "efficientnet":
            label = f"{file.split('-')[0]}, {file.split('-')[1]}"
        else:
            label = f"{file.split('-')[0]}, {file.split('-')[1]}-{file.split('-')[2]}"
        plt.plot(
            df["epoch"],
            df["DiceLoss"],
            label=label,
        )
    plt.legend()
    plt.tight_layout(pad=3)
    # plt.title("Dice Loss / Epoch for UnetPlutPlus model with various encoders")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    fig.savefig(f"figures/model-encoder-comparison-30.png", dpi=1000)


def plot_final_models(csv_dir="final-models"):
    fig = plt.figure(figsize=(9, 8))
    plt.style.use("ggplot")
    for file in sorted(os.listdir(csv_dir)):
        df = pd.read_csv(os.path.join(csv_dir, file))

        if file.split("-")[1] != "efficientnet":
            label = f"{file.split('-')[0]}, {file.split('-')[1]}"
        else:
            label = f"{file.split('-')[0]}, {file.split('-')[1]}-{file.split('-')[2]}"

        plt.plot(df["epoch"], df["DiceLoss"], label=label)
    plt.legend()
    plt.tight_layout(pad=3)
    # plt.title("Dice Loss / Epoch for UnetPlutPlus model with various encoders")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    fig.savefig(f"figures/final-models.png", dpi=1000)


if __name__ == "__main__":
    # plot_model_comp()
    # plot_encoder_comp_unetpp()
    # plot_model_encoder_comp()
    plot_final_models()
