###
### Lifted from SADMA github 
###
'''
UNet Original Paper: Ronneberger et al., 2015 (https://arxiv.org/abs/1505.04597)
UNet Initial PyTorch Implementation: Alexandre Milesi (https://github.com/milesial/Pytorch-UNet)

CBAM Original Paper:
CBAM Initial PyTorch Implementation:

This modified implementation of UNet + CBAM: Azhan Mohammed
Email: azhanmohammed1999@gmail.com
Description: UNet model with Residual Blocks having Channel and Spatial Attention for pixel-level segmentation
'''

import torch
import numpy as np
from torch import nn
import random
from Model.layers import Duck_block
from segmentation_models_pytorch.encoders import get_encoder
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inputChannel, outputChannel, stride)
        self.bn1 = nn.BatchNorm2d(outputChannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outputChannel, outputChannel)
        self.bn2 = nn.BatchNorm2d(outputChannel)
        self.downsample = downsample
        self.ca = ChannelAttention(outputChannel)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        caOutput = self.ca(out)
        out = caOutput * out
        saOutput = self.sa(out)
        out = saOutput * out
        return out, saOutput

class BasicDownSample(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )
    
    def forward(self,x):
        x = self.convolution(x)
        return x

class DownSampleWithAttention(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )
        self.ca = ChannelAttention(outputChannel)
        self.sa = SpatialAttention()
    
    def forward(self,x):
        x = self.convolution(x)
        caOutput = self.ca(x)
        x = caOutput * x
        saOutput = self.sa(x)
        x = saOutput * x
        return x, saOutput

class BasicUpSample(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.convolution(x)
        return x
    
class UpSampleWithAttention(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = ChannelAttention(outputChannel)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.convolution(x)
        caOutput = self.ca(x)
        x = caOutput * x
        saOutput = self.sa(x)
        x = saOutput * x
        return x, saOutput

class UNet(nn.Module):
  def __init__(self, inputChannel, outputChannel):
    super().__init__()
    self.downsample1 = BasicDownSample(inputChannel, 32)
    self.downsample2 = BasicDownSample(32, 64)
    self.downsample3 = BasicDownSample(64, 128)
    self.downsample4 = BasicDownSample(128, 256)
    self.downsample5 = BasicDownSample(256, 512)

    self.upsample1 = BasicUpSample(512, 256)
    self.upsample2 = BasicUpSample(512, 128)
    self.upsample3 = BasicUpSample(256, 64)
    self.upsample4 = BasicUpSample(128, 32)
    self.upsample5 = BasicUpSample(64, 32)
    self.classification = self.classification = nn.Sequential(
            nn.Conv2d(32, outputChannel, kernel_size=1),
        )

  def forward(self, x):
    scale128 = self.downsample1(x)
    scale64 = self.downsample2(scale128)
    scale32 = self.downsample3(scale64)
    scale16 = self.downsample4(scale32)
    scale8 = self.downsample5(scale16)
    upscale16 = self.upsample1(scale8)
    upscale16 = torch.cat([upscale16, scale16], dim=1)
    upscale32 = self.upsample2(upscale16)
    upscale32 = torch.cat([upscale32, scale32], dim=1)
    upscale64 = self.upsample3(upscale32)
    upscale64 = torch.cat([upscale64, scale64], dim=1)
    upscale128 = self.upsample4(upscale64)
    upscale128 = torch.cat([upscale128, scale128], dim=1)
    upscale256 = self.upsample5(upscale128)
    finaloutput = self.classification(upscale256)
    return finaloutput

class AttentionUNet(nn.Module):
  def __init__(self, inputChannel, outputChannel):
    super().__init__()
    self.downsample1 = DownSampleWithAttention(inputChannel, 32)
    self.downsample2 = DownSampleWithAttention(32, 64)
    self.downsample3 = DownSampleWithAttention(64, 128)
    self.downsample4 = DownSampleWithAttention(128, 256)
    self.downsample5 = DownSampleWithAttention(256, 512)

    self.upsample1 = UpSampleWithAttention(512, 256)
    self.upsample2 = UpSampleWithAttention(512, 128)
    self.upsample3 = UpSampleWithAttention(256, 64)
    self.upsample4 = UpSampleWithAttention(128, 32)
    self.upsample5 = UpSampleWithAttention(64, 32)
    self.classification = nn.Sequential(
            nn.Conv2d(32, outputChannel, kernel_size=1),
        )

  def forward(self, x):
    scale128, sa128down = self.downsample1(x)
    scale64, sa64down = self.downsample2(scale128)
    scale32, sa32down = self.downsample3(scale64)
    scale16, sa64down = self.downsample4(scale32)
    scale8, sa8down = self.downsample5(scale16)
    upscale16, sa16up = self.upsample1(scale8)
    upscale16 = torch.cat([upscale16, scale16], dim=1)
    upscale32, sa32up = self.upsample2(upscale16)
    upscale32 = torch.cat([upscale32, scale32], dim=1)
    upscale64, sa64up = self.upsample3(upscale32)
    upscale64 = torch.cat([upscale64, scale64], dim=1)
    upscale128, sa128up = self.upsample4(upscale64)
    upscale128 = torch.cat([upscale128, scale128], dim=1)
    upscale256, sa256up = self.upsample5(upscale128)
    finaloutput = self.classification(upscale256)
    return finaloutput

class ResidualAttentionUNet(nn.Module):
  def __init__(self, inputChannel, outputChannel):
    super().__init__()
    self.downsample1 = DownSampleWithAttention(inputChannel, 32)
    self.downsample2 = DownSampleWithAttention(32, 64)
    self.downsample3 = DownSampleWithAttention(64, 128)
    self.downsample4 = DownSampleWithAttention(128, 256)
    self.downsample5 = DownSampleWithAttention(256, 512)

    self.residualBlock1 = ResidualBlock(512, 512)
    self.residualBlock2 = ResidualBlock(512, 512)
    self.residualBlock3 = ResidualBlock(512, 512)

    self.upsample1 = UpSampleWithAttention(512, 256)
    self.upsample2 = UpSampleWithAttention(512, 128)
    self.upsample3 = UpSampleWithAttention(256, 64)
    self.upsample4 = UpSampleWithAttention(128, 32)
    self.upsample5 = UpSampleWithAttention(64, 32)
    self.classification = nn.Sequential(
            nn.Conv2d(32, outputChannel, kernel_size=1),
        )

  def forward(self, x):
    scale128, sa128down = self.downsample1(x)
    scale64, sa64down = self.downsample2(scale128)
    scale32, sa32down = self.downsample3(scale64)
    scale16, sa64down = self.downsample4(scale32)
    scale8, sa8down = self.downsample5(scale16)
    scale8, sa8down = self.residualBlock1(scale8)
    scale8, sa8down = self.residualBlock2(scale8)
    scale8, sa8down = self.residualBlock3(scale8)
    upscale16, sa16up = self.upsample1(scale8)
    upscale16 = torch.cat([upscale16, scale16], dim=1)
    upscale32, sa32up = self.upsample2(upscale16)
    upscale32 = torch.cat([upscale32, scale32], dim=1)
    upscale64, sa64up = self.upsample3(upscale32)
    upscale64 = torch.cat([upscale64, scale64], dim=1)
    upscale128, sa128up = self.upsample4(upscale64)
    upscale128 = torch.cat([upscale128, scale128], dim=1)
    upscale256, sa256up = self.upsample5(upscale128)
    finaloutput = self.classification(upscale256)
    return finaloutput
  
class ResidualAttentionDuckUNet(nn.Module):
  def __init__(self, inputChannel, outputChannel):
    super().__init__()
    self.downsample1 = DownSampleWithAttention(inputChannel, 32)
    self.downsample2 = DownSampleWithAttention(32, 64)
    self.downsample3 = DownSampleWithAttention(64, 128)
    self.downsample4 = DownSampleWithAttention(128, 256)
    self.downsample5 = DownSampleWithAttention(256, 512)

    self.residualBlock1 = ResidualBlock(512, 512)
    self.residualBlock2 = ResidualBlock(512, 512)
    self.residualBlock3 = ResidualBlock(512, 512)

    self.upsample1 = UpSampleWithAttention(512, 256)
    self.upsample2 = UpSampleWithAttention(512, 128)
    self.upsample3 = UpSampleWithAttention(256, 64)
    self.upsample4 = UpSampleWithAttention(128, 32)
    self.upsample5 = UpSampleWithAttention(64, 32)
    self.classification = nn.Sequential(
            nn.Conv2d(32, outputChannel, kernel_size=1),
        )

    self.duck1 = Duck_block(32, 32)
    self.duck2 = Duck_block(64, 64)
    self.duck3 = Duck_block(128, 128)
    self.duck4 = Duck_block(256, 256)
    self.duck5 = Duck_block(512, 512)
    self.duck1up = Duck_block(512, 512)
    self.duck2up = Duck_block(256, 256)
    self.duck3up = Duck_block(128, 128)
    self.duck4up = Duck_block(64, 64)
    self.duck5up = Duck_block(32, 32)


  def forward(self, x):
    scale128, sa128down = self.downsample1(x)
    d_scale128 = self.duck1(scale128)
    scale64, sa64down = self.downsample2(scale128 + d_scale128)
    d_scale64 = self.duck2(scale64)
    scale32, sa32down = self.downsample3(scale64 + d_scale64)
    d_scale32 = self.duck3(scale32)
    scale16, sa64down = self.downsample4(scale32 + d_scale32)
    d_scale16 = self.duck4(scale16)
    scale8, sa8down = self.downsample5(scale16 + d_scale16)
    d_scale8 = self.duck5(scale8)
    scale8, sa8down = self.residualBlock1(scale8 + d_scale8)
    scale8, sa8down = self.residualBlock2(scale8)
    scale8, sa8down = self.residualBlock3(scale8)
    upscale16, sa16up = self.upsample1(scale8)
    upscale16_d = self.duck1up(upscale16)
    upscale16 = torch.cat([upscale16 + upscale16_d, d_scale16], dim=1)
    upscale32, sa32up = self.upsample2(upscale16)
    upscale32_d = self.duck2up(upscale32)
    upscale32 = torch.cat([upscale32 + upscale32_d, d_scale32], dim=1)
    upscale64, sa64up = self.upsample3(upscale32)
    upscale64_d = self.duck3up(upscale64)
    upscale64 = torch.cat([upscale64+ upscale64_d, d_scale64], dim=1)
    upscale128, sa128up = self.upsample4(upscale64)
    upscale128_d = self.duck4up(upscale128)
    upscale128 = torch.cat([upscale128 + upscale128_d, d_scale128], dim=1)
    upscale256, sa256up = self.upsample5(upscale128)
    finaloutput = self.classification(upscale256)
    return finaloutput


class ResidualAttentionDuckNetwithEncoder(nn.Module):
  def __init__(self, inputChannel, outputChannel):
    super().__init__()
    self.downsample1 = DownSampleWithAttention(inputChannel, 16)
    self.downsample2 = DownSampleWithAttention(64, 64)
    self.downsample3 = DownSampleWithAttention(104, 128)
    self.downsample4 = DownSampleWithAttention(192, 256)
    self.downsample5 = DownSampleWithAttention(432, 512)

    self.residualBlock1 = ResidualBlock(512, 512)
    self.residualBlock2 = ResidualBlock(512, 512)
    self.residualBlock3 = ResidualBlock(512, 512)

    self.upsample1 = UpSampleWithAttention(512, 256)
    self.upsample2 = UpSampleWithAttention(512, 128)
    self.upsample3 = UpSampleWithAttention(256, 64)
    self.upsample4 = UpSampleWithAttention(128, 32)
    self.upsample5 = UpSampleWithAttention(64, 32)
    self.classification = nn.Sequential(
            nn.Conv2d(32, outputChannel, kernel_size=1),
        )
    
    self.duckCH = Duck_block(1024, 512)
    self.duck1down = Duck_block(16, 16)
    self.duck2down = Duck_block(64, 64)
    self.duck3down = Duck_block(128, 128)
    self.duck4down = Duck_block(256, 256)
    self.duck5down = Duck_block(512, 512)
    self.duck1up = Duck_block(512, 512)
    self.duck2up = Duck_block(256, 256)
    self.duck3up = Duck_block(128, 128)
    self.duck4up = Duck_block(64, 64)
    self.duck5up = Duck_block(32, 48)

    
    self.encoder = get_encoder("efficientnet-b5", weights="imagenet", in_channels=3)

  def forward(self, x):
    encoded = self.encoder(x)
    scale128, sa128down = self.downsample1(encoded[0])
    #print("scale128:", scale128.shape)
    d_scale128 = self.duck1down(scale128)
    scale64, sa64down = self.downsample2(torch.cat([encoded[1],  d_scale128 + scale128], dim=1))
    #print("scale64:", scale64.shape)
    d_scale64 = self.duck2down(scale64)
    scale32, sa32down = self.downsample3(torch.cat([encoded[2],  d_scale64 + scale64], dim=1))
    #print("scale32:", scale32.shape)
    d_scale32 = self.duck3down(scale32)
    scale16, sa64down = self.downsample4(torch.cat([encoded[3], d_scale32 + scale32], dim=1))
    d_scale16 = self.duck4down(scale16)
    scale8, sa8down = self.downsample5(torch.cat([encoded[4],  d_scale16 + scale16], dim=1))
    d_scale8 = self.duck5down(scale8)
    scale8_cat = torch.cat([encoded[5], d_scale8 + scale8], dim=1)
    scale8_cat = self.duckCH(scale8_cat)
    scale8, sa8down = self.residualBlock1(scale8_cat)
    scale8, sa8down = self.residualBlock2(scale8)

    scale8, sa8down = self.residualBlock3(scale8)
    scale8_d = self.duck1up(scale8)
    upscale16, sa16up = self.upsample1(scale8_d)
    upscale16_d = self.duck2up(upscale16)
    upscale16 = torch.cat([upscale16_d + upscale16, d_scale16], dim=1)
    upscale32, sa32up = self.upsample2(upscale16)
    upscale32_d = self.duck3up(upscale32)
    upscale32 = torch.cat([upscale32_d + upscale32, d_scale32], dim=1)
    upscale64, sa64up = self.upsample3(upscale32)
    upscale64_d = self.duck4up(upscale64)
    upscale64 = torch.cat([upscale64_d + upscale64, d_scale64], dim=1)
    upscale128, sa128up = self.upsample4(upscale64)
    upscale128_d = self.duck5up(upscale128)
    upscale128 = torch.cat([upscale128_d, d_scale128], dim=1)
    upscale256, sa256up = self.upsample5(upscale128)
    finaloutput = self.classification(upscale256)
    return finaloutput
   

# Shapes:
# scale128: torch.Size([1, 32, 256, 256])
# scale64: torch.Size([1, 64, 128, 128])
# scale32: torch.Size([1, 128, 64, 64])
# scale16: torch.Size([1, 256, 32, 32])
# scale8: torch.Size([1, 512, 16, 16])
# upscale16: torch.Size([1, 512, 32, 32])
# upscale32: torch.Size([1, 256, 64, 64])
# upscale64: torch.Size([1, 128, 128, 128])
# upscale128: torch.Size([1, 64, 256, 256])
# upscale256: torch.Size([1, 32, 512, 512])
# finaloutput: torch.Size([1, 1, 512, 512])
# torch.Size([1, 1, 512, 512])
