import torch 
import torch.nn as nn
import collections

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Convformer(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self
        self.separable_conv = SeparableConv2d(in_channels, out_channels, kernel_size=3)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.3)
        self.out_channels = out_channels
        self.in_channels = in_channels
        
    
    def forward(self, x):
        input = x
        x = nn.LayerNorm()(x)
        x = self.separable_conv(x)
        x = self.attention(x, x, x)
        x = x + input
        x1 = nn.Linear(self.out_channels)(x)
        x1 = nn.GELU()(x1)
        x1 = nn.Linear(self.out_channels)(x1)
        return x1 + x
    
class Bn_act(nn.Module):
    def __init__(self, activation=nn.SiLU(), channels=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm2d(channels)
        self.act = activation
    
    def forward(self, x):
        if self.act is None:
            return self.bn(x)
        return self.act(self.bn(x))
    
class Decode(nn.Module):
    def __init__(self,  out_channels, in_channels=4, scale=2, activation=nn.ReLU(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.scale = scale
        self.activation = activation
        self.out_channels = out_channels
        self.bn_act = Bn_act(activation, out_channels)
    
    def forward(self, input):
        x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)(input)
        x1 = self.activation(x1)
        x2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3)(input)
        x2 = self.activation(x2)

        merge = x1 + x2
        x = nn.UpsamplingNearest2d(scale_factor=self.scale)(merge)

        skip_feature = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3)(merge)
        skip_feature = self.activation(skip_feature)
        skip_feature = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)(skip_feature)
        skip_feature = self.activation(skip_feature)
        
        merge = merge + skip_feature
        return self.bn_act(merge)
    

class Merge(nn.Module):
    def __init__(self, out_channels=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
    
    def forward(self, input):
        if self.out_channels is None:
            return input
        
        
class Scoped_conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mode="wide", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.mode = mode
        assert self.mode in ["wide", "mid"]
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channels),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding='same', dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channels),
        )
        self.wide = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding='same', dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channels),
        )
        

    def forward(self, input):
        x = self.block(input)
        if self.mode == "wide":
            x = self.wide(x)
        return x


class ResNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, input):
        x = self.block(input)
        x = x + self.shortcut(input)
        return x
    

class Duck_block(nn.Module):
    def __init__(self, in_channels, out_channels, size=6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.size = size
        self.widescope = Scoped_conv2D(self.in_channels, self.out_channels, mode="wide")
        self.midscope = Scoped_conv2D(self.in_channels, self.out_channels, mode="mid")
        self.conv1 = nn.Sequential(
            ResNet_block(self.in_channels, self.out_channels, dilation=1),
        )
        self.conv2 = nn.Sequential(
            ResNet_block(self.in_channels, self.out_channels, dilation=1),
            ResNet_block(self.out_channels, self.out_channels, dilation=1),
        )
        self.conv3 = nn.Sequential(
            ResNet_block(self.in_channels, self.out_channels, dilation=1),
            ResNet_block(self.out_channels, self.out_channels, dilation=1),
            ResNet_block(self.out_channels, self.out_channels, dilation=1),
        )
        self.seperate = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, size), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channels),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(size, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channels),
        )
        self.out_norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, input):
        x1 = self.widescope(input)
        x2 = self.midscope(input)
        x3 = self.conv1(input)
        x4 = self.conv2(input)
        x5 = self.conv3(input)
        x6 = self.seperate(input)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.out_norm(x)
        return x


