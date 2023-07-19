import torch
import os
import subprocess
import click

import segmentation_models_pytorch as smp
from torchvision import transforms
from Dataset.dataset import ETHDataset
from PIL import Image
from Model.model import ResidualAttentionUNet, ResidualAttentionDuckUNet
from torchvision.utils import save_image


@click.command()
@click.option(
    "--model-name",
    "-m",
    help="Which model to use",
    default="UnetPlusPlus",
    type=str,
)
@click.option(
    "--encoder-name",
    "-en",
    help="Which encoder to use",
    default="resnet34",
    type=str,
)
@click.option(
    "--encoder-weights",
    "-ew",
    help="Which encoder weights to use",
    default="imagenet",
    type=str,
)
@click.option(
    "--loss",
    "-l",
    help="Which loss was used",
    default="clDice",
    type=str,
)
def main(model_name, encoder_name, encoder_weights, loss):
    base_path = "data/ethz-cil-road-segmentation-2023"
    image_path = os.path.join(base_path, "test/images")
    pad = transforms.Compose([transforms.Pad(56, fill=(0)), transforms.ToTensor()])

    if model_name == "ResidualAttentionUNet":
        model = ResidualAttentionUNet(4, 1)
    else:
        model_func = getattr(smp, model_name)

        model = model_func(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            # activation="sigmoid",
            in_channels=4,
            classes=1,
        )
    model.load_state_dict(
        torch.load(
            f"models/model_weights_{model_name}-{encoder_name}-{encoder_weights}-{loss}.pth",
            map_location=torch.device("cpu"),
        )
    )
    model.eval()
    for file in os.listdir(image_path):
        print(file)
        image = Image.open(os.path.join(image_path, file))
        image = pad(image)
        mask = model(image.unsqueeze(0))
        mask = transforms.CenterCrop(400)(mask)
        save_image(mask.squeeze(0), f"submission/masks/{file}")


if __name__ == "__main__":
    main()
