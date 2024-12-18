import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Contracting Path with 4 encoder blocks and 4 max pooling layers
        self.encoder1_conv = self.conv_ReLu(in_channels, 64)
        self.encoder1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2_conv = self.conv_ReLu(64, 128)
        self.encoder2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3_conv = self.conv_ReLu(128, 256)
        self.encoder3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4_conv = self.conv_ReLu(256, 512)
        self.encoder4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Base Layer with 2 convolutional layers
        self.base_layer = self.conv_ReLu(512, 1024)

        # Expansive Path with 4 decoder blocks and 4 up-convolutional layers
        self.decoder1_convT = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1_conv = self.conv_ReLu(1024, 512)

        self.decoder2_convT = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2_conv = self.conv_ReLu(512, 256)

        self.decoder3_convT = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3_conv = self.conv_ReLu(256, 128)

        self.decoder4_convT = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4_conv = self.conv_ReLu(128, 64)

        # Final Convolutional Layer, 3 output channels because it is a RGB image
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=1, bias=True)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

        self.weight_init()

    def conv_ReLu(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Convolutional block with ReLU activation function, Batch Normalization and 2 convolutional layers.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            nn.Sequential: Convolutional block with ReLU activation function, Batch Normalization and 2 convolutional layers.
        """
        return  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
   
        )

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, X):
        s1 = self.encoder1_conv(X)
        p1 = self.encoder1_pool(s1)

        s2 = self.encoder2_conv(p1)
        p2 = self.encoder2_pool(s2)

        s3 = self.encoder3_conv(p2)
        p3 = self.encoder3_pool(s3)

        s4 = self.encoder4_conv(p3)
        p4 = self.encoder4_pool(s4)

        base = self.base_layer(p4)

        # Concatenation of the skip connections
        d4 = self.decoder1_convT(base)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.decoder1_conv(d4)

        d3 = self.decoder2_convT(d4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.decoder2_conv(d3)

        d2 = self.decoder3_convT(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.decoder3_conv(d2)

        d1 = self.decoder4_convT(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.decoder4_conv(d1)

        res = self.final_conv(d1)
        return res

    def predict(self, X):
        return self.sigmoid(self.forward(X))
