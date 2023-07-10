from trainer import Trainer
from Model.model import ResidualAttentionUNet, AttentionUNet, UNet
from Model.custom_unet import CustomUnet
from Dataset.dataset import ETHDataset, MassachusettsDataset, GMapsDataset
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from Loss.cldice import soft_cldice, soft_dice_cldice
import os
import torch
import torchvision

if __name__ == "__main__":
    # base_path = "data/ethz-cil-road-segmentation-2023"
    # image_path = os.path.join(base_path, "training/images")
    # mask_path = os.path.join(base_path, "training/groundtruth")
    # dataset = ETHDataset(image_path, mask_path, augment_images=False)

    # dataset = MassachusettsDataset('data/archive/tiff/train', 'data/archive/tiff/train_labels', augment_images=False)

    base_path = "data/additional_data"
    image_path = os.path.join(base_path, "images")
    mask_path = os.path.join(base_path, "masks")
    dataset = GMapsDataset(image_path, mask_path, augment_images=True)

    # model = ResidualAttentionUNet(4, 1)
    # model.load_state_dict(torch.load('model_weights_2023-07-05_15.pth'))

    fe = torchvision.models.mobilenet_v3_large()
    fe.load_state_dict(torch.load('best_model_feature_extractor.pth'))
    fe.classifier = torch.nn.Identity()
    model = CustomUnet(fe)

    loss_fn = DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainer = Trainer(model, dataset, None, loss_fn, None, split_test=0.2, batch_size=4, epochs=20, test_metrics=[JaccardLoss(mode='binary'), loss_fn], test_metric_names=["JaccardLoss", "DiceLoss"], epochs_between_safe=10, name="pretrained")
    scores = trainer.train_test()
    scores.to_csv("test_scores_pretrained.csv")
