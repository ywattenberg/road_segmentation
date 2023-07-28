import torch
import numpy as np
import pandas as pd
import os
import click
from Dataset.dataset import ETHDataset
from torch.utils.data import DataLoader
from Model.model import ResidualAttentionDuckNetwithEncoder
import segmentation_models_pytorch as smp


def get_model(model_name, encoder_name, encoder_weights, model_path, device=None):
    model_func = getattr(smp, model_name)

    model = model_func(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        # activation="sigmoid",
        in_channels=3,
        classes=1,
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if device is not None:
        model.to(device)
    return model


def calc_scores(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).to(torch.int32)
            outputs = model(images)
            if i == 0:
                all_outputs = outputs
                all_labels = labels
            else:
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
        all_outputs = all_outputs.squeeze(dim=1)
        print(all_outputs.shape)
        print(all_labels.shape)
        tp, fp, fn, tn = smp.metrics.get_stats(
            all_outputs, all_labels, mode="binary", threshold=0.5
        )

        # then compute metrics with required reduction (see metric docs)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        results = [iou_score, f1_score, f2_score, accuracy, recall]
        results = [result.cpu().numpy() for result in results]

    return results


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
    default="efficientnet-b5",
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
    "--best-model",
    is_flag=True,
    default=False,
    help="Whether to use the best model",
)
@click.option("--output_path", default="scores.csv", help="Path to write results")
def main(model_name, encoder_name, encoder_weights, best_model, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if best_model:
        model_path = f"models/best_model_weights_{model_name}-{encoder_name}-{encoder_weights}-clDice.pth"
    else:
        model_path = f"models/model_weights_{model_name}-{encoder_name}-{encoder_weights}-clDice.pth"
    print(f"Using model: {model_path}")

    model = get_model(
        model_name, encoder_name, encoder_weights, model_path, device=device
    )
    model.to(device)
    base_path = "data/ethz-cil-road-segmentation-2023/training"
    dataset = ETHDataset(
        os.path.join(base_path, "images"),
        os.path.join(base_path, "groundtruth"),
        augment_images=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if os.path.exists(output_path):
        scores = pd.read_csv(output_path)
    else:
        scores = pd.DataFrame(
            columns=["model", "iou_score", "f1_score", "f2_score", "accuracy", "recall"]
        )

    scores.loc[len(scores)] = [model_path] + calc_scores(model, dataloader, device)
    scores.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
