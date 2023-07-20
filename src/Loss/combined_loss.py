import torch
from torch import nn
from segmentation_models_pytorch import losses

class DiceLovaszBCELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.03, gamma=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dice = losses.DiceLoss(mode="binary")
        self.lovasz = losses.LovaszLoss(mode="binary")
        self.BCE = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pr, y_gt):
        dice = self.dice(y_pr, y_gt)
        lovasz = self.lovasz(y_pr, y_gt)
        bce = self.BCE(y_pr, y_gt)
        #print(f"dice: {dice}, lovasz: {lovasz}, bce: {bce}")
        return  self.alpha * dice + self.beta * lovasz + self.gamma * bce
    