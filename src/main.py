from trainer import Trainer
from Model.model import ResidualAttentionUNet, AttentionUNet, UNet, ResidualAttentionDuckUNet
from lion_pytorch import Lion
from Dataset.dataset import ETHDataset, MassachusettsDataset, GMapsDataset 
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from Loss.cldice import SoftDiceClDice
import os
import torch
import torchvision
import sys

if __name__ == "__main__":\
    # use line-buffering for both stdout and stderr
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
    # base_path = "data/ethz-cil-road-segmentation-2023"
    # image_path = os.path.join(base_path, "training/images")
    # mask_path = os.path.join(base_path, "training/groundtruth")
    # dataset = ETHDataset(image_path, mask_path, augment_images=False)

    # dataset = MassachusettsDataset('data/archive/tiff/train', 'data/archive/tiff/train_labels', augment_images=False)

    base_path = "data/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    skeleton_path = os.path.join(base_path, "skel")
    dataset = GMapsDataset(image_path, mask_path, skel_path=skeleton_path, augment_images=True, normalize=True)

    model = ResidualAttentionDuckUNet(4, 1)
    # model.load_state_dict(torch.load('model_weights_2023-07-05_15.pth'))

    loss_fn = SoftDiceClDice(0.5)
    optimizer = Lion(model.parameters(), lr=1e-3, weight_decay=1e-3)
    trainer = Trainer(model, dataset, None, loss_fn, None, split_test=0.2, batch_size=64, epochs=60, test_metrics=[JaccardLoss(mode='binary'), DiceLoss(mode='binary'), loss_fn], test_metric_names=["JaccardLoss", "DiceLoss", "clDice"], epochs_between_safe=1, name="pretrained_clDice_duck")
    scores = trainer.train_test()
    scores.to_csv("test_scores_clDice_duck.csv")
