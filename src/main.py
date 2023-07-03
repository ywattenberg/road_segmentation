from trainer import Trainer
from Model.unet import UNet
from Dataset.dataset import ETHDataset
import os
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss


if __name__ == "__main__":
    base_path = "data/ethz-cil-road-segmentation-2023"
    image_path = os.path.join(base_path, "training/images")
    mask_path = os.path.join(base_path, "training/groundtruth")
    dataset = ETHDataset(image_path, mask_path, augment_images=True)
    model = UNet(4, 1)
    loss_fn = DiceLoss(mode='binary')
    trainer = Trainer(model, dataset, None, loss_fn, None, split_test=0.2, batch_size=8, epochs=10, test_metrics=[loss_fn, JaccardLoss(mode='binary')])
    trainer.train_test()
