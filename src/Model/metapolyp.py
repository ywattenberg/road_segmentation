import torch
from torch import nn
from layers import Convformer
import torchvision

class MetaPolypModel(nn.Module):
    def __init__(self, img_size=400, in_channels=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.in_channels = in_channels
        self.backbone = torchvision.models.vit_b_16(pretrained=True)
        

