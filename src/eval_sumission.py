import segmentation_models_pytorch as smp
import os
import sys
import torch
from Dataset.dataset import ETHDataset
import click


@click.command()
@click.option("--base_dir", default="submission", help="Directory with predicted masks")
def main(base_dir):
    base_path = "data/ethz-cil-road-segmentation-2023"
    #image_path = os.path.join(base_path, "training/images")
    mask_path = os.path.join(base_path, "training/groundtruth")
    dataset = ETHDataset(mask_path, augment_images=False, submission=True)
    masks = ETHDataset(base_dir, augment_images=False, submission=True)
    output = torch.cat([mask for _,mask,_ in masks], dim=0)
    target = torch.cat([mask for _,mask,_ in dataset], dim=0) 
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='binary', threshold=0.5)

    # then compute metrics with required reduction (see metric docs)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    

if __name__ == "__main__":  # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()
