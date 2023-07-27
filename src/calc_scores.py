import torch
import numpy as np
import pandas as pd
import os
import click
from Dataset.dataset import ETHDataset
from torch.utils.data import DataLoader
from Model.model import ResidualAttentionDuckNetwithEncoder
import segmentation_models_pytorch as smp


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

def calc_scores(model, dataloader, device):

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if i == 0:
                all_outputs = outputs
                all_labels = labels
            else:
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
            print (all_outputs.shape)
            print (all_labels.shape)
            tp, fp, fn, tn = smp.metrics.get_stats(all_outputs, all_labels, mode='binary', threshold=0.5)

            # then compute metrics with required reduction (see metric docs)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    return [iou_score, f1_score, f2_score, accuracy, recall]

@click.command()
@click.option('--model_path', default='Model/weights/ResidualAttentionDuckNetwithEncoder.pth', help='Path to model weights')
@click.option('--output_path', default='scores.csv', help='Path to write results')
@click.option('--is_smp', default=False, help='Whether to use segmentation_models_pytorch')
def main(model_path, output_path, is_smp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_smp:
        model = os.path.split("_")[-1]
        model, encoder = model.split("-")[0:2]
        print(f"Using model {model} with encoder {encoder}")
        model = get_model(model, encoder, model_path, device=device)
    else:
        model = ResidualAttentionDuckNetwithEncoder(3,1)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    base_path = "data/ethz-cil-road-segmentation-2023/training"
    dataset = ETHDataset(os.path.join(base_path, "images"), os.path.join(base_path, "groundtruth"), augment_images=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if os.path.exists(output_path):
        scores = pd.read_csv(output_path)
    else:
        scores = pd.DataFrame(columns=['model','iou_score', 'f1_score', 'f2_score', 'accuracy', 'recall'])
    
    scores.loc[len(scores)] = [model_path] + calc_scores(model, dataloader, device)
    scores.to_csv(output_path, index=False)
