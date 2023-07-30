from trainer import Trainer
from Model.model import (
    ResidualAttentionUNet,
    AttentionUNet,
    UNet,
    ResidualAttentionDuckUNet,
    ResidualAttentionDuckNetwithEncoder
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
import click

@click.command()
@click.option(
    "--tmpdir",
    "-t",
    type=str,
)
def main(tmpdir):
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    torch.manual_seed(0)
    # base_path = "data/ethz-cil-road-segmentation-2023"
    # image_path = os.path.join(base_path, "training/images")
    # mask_path = os.path.join(base_path, "training/groundtruth")
    # test_dataset = ETHDataset(image_path, mask_path, augment_images=False)

    # dataset = MassachusettsDataset('data/archive/tiff/train', 'data/archive/tiff/train_labels', augment_images=False)

    base_path = f"{tmpdir}/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    skeleton_path = os.path.join(base_path, "skel")
    dataset = GMapsDataset(
        image_path,
        mask_path,
        skel_path=skeleton_path,
        augment_images=True,
        #normalize=True,
    )

    model =  ResidualAttentionDuckNetwithEncoder(3, 1)
    #model.load_state_dict(torch.load('best_model_weights_pretrained_clDice_duck_final.pth'))

    loss_fn = SoftDiceClDice(0.5) #DiceLovaszBCELoss(alpha=0.5, beta=0.3, gamma=0.2)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-6)
    trainer = Trainer(
        model,
        dataset,
        None,
        loss_fn,
        optimizer,
        split_test=0.2,
        batch_size=16,
        epochs=10,
        test_metrics=[DiceLoss(mode="binary"), JaccardLoss(mode="binary")],
        test_metric_names=["DiceLoss", "JaccardLoss"],
        epochs_between_safe=1,
        name="our_model_cl_dice",
        
    )
    scores = trainer.train_test()
    scores.to_csv("our_model_cl_dice.csv")

if __name__ == "__main__":  # use line-buffering for both stdout and stderr
    main()
