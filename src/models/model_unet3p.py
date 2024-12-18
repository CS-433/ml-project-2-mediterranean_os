import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50


class conv_block(nn.Module):
    def __init__(self, in_c,out_c,act=True):
        super().__init__()

        layers = [nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1)]

        if act == True:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            conv_block(in_c, out_c),
            conv_block(out_c, out_c)
        )
        self.p1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        p = self.p1(x)
        return x, p

class unet3p(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.input_size = input_size
        size = input_size
        while(size % 128 != 0):
            size = size + 1 
        padding_size = (int) ((size - input_size)/2)
        self.pad = nn.ConstantPad2d(padding_size, value=0)  

        """ Encoder """
        self.encoder1 = encoder_block(3, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)

        """ Bottleneck """
        self.e5 = nn.Sequential(
            conv_block(512, 1024),
            conv_block(1024, 1024)
        )

        """ Decoder 4 """
        self.e1_d4 = conv_block(64, 64)
        self.e2_d4 = conv_block(128, 64)
        self.e3_d4 = conv_block(256, 64)
        self.e4_d4 = conv_block(512, 64)
        self.e5_d4 = conv_block(1024, 64)

        self.d4 = conv_block(64*5, 64)

        """ Decoder 3 """
        self.e1_d3 = conv_block(64, 64)
        self.e2_d3 = conv_block(128, 64)
        self.e3_d3 = conv_block(256, 64)
        self.e4_d3 = conv_block(64, 64)
        self.e5_d3 = conv_block(1024, 64)

        self.d3 = conv_block(64*4, 64)

        """ Decoder 2 """
        self.e1_d2 = conv_block(64, 64)
        self.e2_d2 = conv_block(128, 64)
        self.e3_d2 = conv_block(64, 64)
        self.e4_d2 = conv_block(64, 64)
        self.e5_d2 = conv_block(1024, 64)

        self.d2 = conv_block(64*3, 64)

        """ Decoder 1 """
        self.e1_d1 = conv_block(64, 64)
        self.e2_d1 = conv_block(64, 64)
        self.e3_d1 = conv_block(64, 64)
        self.e4_d1 = conv_block(64, 64)
        self.e5_d1 = conv_block(1024, 64)

        self.d1 = conv_block(64*2, 64)

        self.d0 = conv_block(64*2, 64)

        self.y1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        """ Encoder """
        e1, p1 = self.encoder1(inputs)
        e2, p2 = self.encoder2(p1)
        e3, p3 = self.encoder3(p2)
        e4, p4 = self.encoder4(p3)

        """ Bottleneck """
        e5 = self.e5(e4)

        """ Decoder 4 """
        e1_d4 = F.max_pool2d(e1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(e2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(e3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(e4)

        e5_d4 = self.e5_d4(e5)

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)

        d4 = self.d4(d4)

        """ Decoder 3 """
        e1_d3 = F.max_pool2d(e1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        d4 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        
        d3 = torch.cat([e1_d3, e2_d3, e3_d3, d4], dim=1)
        
        d3 = self.d3(d3)

        """ Decoder 2 """
        e1_d2 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        d3 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)

        d2 = torch.cat([e1_d2, e2_d2, d3], dim=1)
        d2 = self.d2(d2)

        """ Decoder 1 """
        e1_d1 = self.e1_d1(e1)

        d2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)

        d1 = torch.cat([e1_d1, d2], dim=1)
        d1 = self.d1(d1)
    
        """ Output """
        out = self.y1(d1)

        return out