import torch
from torch import nn
import torchvision
from Model.model import ResidualAttentionUNet

class CustomUnet(nn.Module):
    def __init__(self, feature_extraction_model=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature_extraction_model = feature_extraction_model
        # self.feature_extraction_model.eval()
        # self.feature_extraction_model.requires_grad_(False)
        self.unet = ResidualAttentionUNet(4,1)

    
    def forward(self, x):
        x = self.feature_extraction_model(x)
        return self.unet(x)
        
        