from trainer import Trainer
from Model.model import ResidualAttentionUNet, AttentionUNet, UNet
from Dataset.dataset import ETHDataset, MassachusettsDataset, GMapsDataset
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import os
import torch

if __name__ == "__main__":
    # base_path = "data/ethz-cil-road-segmentation-2023"
    # image_path = os.path.join(base_path, "training/images")
    # mask_path = os.path.join(base_path, "training/groundtruth")
    # dataset = ETHDataset(image_path, mask_path, augment_images=False)
    #dataset = MassachusettsDataset('data/archive/tiff/train', 'data/archive/tiff/train_labels', augment_images=True)
    base_path = "data/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    dataset = GMapsDataset(image_path, mask_path, augment_images=False)
    model = ResidualAttentionUNet(4, 1)
    model.load_state_dict(torch.load('model_weights_2023-07-05_10.pth'))
    loss_fn = DiceLoss(mode='binary')
    trainer = Trainer(model, dataset, None, loss_fn, None, split_test=0.2, batch_size=4, epochs=5, test_metrics=[JaccardLoss(mode='binary'), loss_fn])
    trainer.train_test()
