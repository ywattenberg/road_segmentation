from trainer import Trainer
from Model.model import (
    ResidualAttentionUNet,
    AttentionUNet,
    UNet,
    ResidualAttentionDuckUNet,
)
from lion_pytorch import Lion
from Dataset.dataset import ETHDataset, MassachusettsDataset, GMapsDataset 
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from Loss.cldice import SoftDiceClDice
from Loss.combined_loss import DiceLovaszBCELoss
import os
import torch
import torchvision
import sys

if __name__ == "__main__":  # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    # base_path = "data/ethz-cil-road-segmentation-2023"
    # image_path = os.path.join(base_path, "training/images")
    # mask_path = os.path.join(base_path, "training/groundtruth")
    # dataset = ETHDataset(image_path, mask_path, augment_images=False)

    # dataset = MassachusettsDataset('data/archive/tiff/train', 'data/archive/tiff/train_labels', augment_images=False)

    base_path = "data/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    skeleton_path = os.path.join(base_path, "skel")
    dataset = GMapsDataset(
        image_path,
        mask_path,
        skel_path=skeleton_path,
        augment_images=True,
        normalize=True,
    )

    model = ResidualAttentionDuckUNet(4, 1)
    model.load_state_dict(torch.load('best_model_weights_pretrained_clDice_duck_final.pth'))

    loss_fn = DiceLovaszBCELoss(alpha=0.5, beta=0.5)
    optimizer = Lion(model.parameters(), lr=1e-5, weight_decay=1e-6)
    trainer = Trainer(
        model,
        dataset,
        None,
        loss_fn,
        None,
        split_test=0.2,
        batch_size=32,
        epochs=10,
        test_metrics=[DiceLoss(mode="binary"), JaccardLoss(mode="binary"),  loss_fn],
        test_metric_names=["DiceLoss", "JaccardLoss", "DiceLovaszBCELoss"],
        epochs_between_safe=1,
        name="test+loss",
    )
    scores = trainer.train_test()
    scores.to_csv("test+loss.csv")
