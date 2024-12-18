import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DecoderBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, conv_in_channels, conv_out_channels, up_in_channels=None, up_out_channels=None):
        super().__init__()
        """
        eg:
        decoder1:
        up_in_channels      : 1024,     up_out_channels     : 512
        conv_in_channels    : 1024,     conv_out_channels   : 512

        decoder5:
        up_in_channels      : 64,       up_out_channels     : 64
        conv_in_channels    : 128,      conv_out_channels   : 64
        """
        if up_in_channels==None:
            up_in_channels=conv_in_channels
        if up_out_channels==None:
            up_out_channels=conv_out_channels

        self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3)
        )

    # x1-upconv , x2-downconv
    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UnetResnet34(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()

        self.input_size = input_size
        size = input_size
        while(size % 64 != 0):
            size = size + 1
        padding_size = (int) ((size - input_size)/2)
        self.pad = nn.ConstantPad2d(padding_size, value=0)  

        resnet34 = torchvision.models.resnet34(pretrained=True)
        filters = [64, 128, 256, 512]

        self.firstlayer = nn.Sequential(*list(resnet34.children())[:3])
        self.maxpool = list(resnet34.children())[3]
        self.encoder1 = resnet34.layer1
        self.encoder2 = resnet34.layer2
        self.encoder3 = resnet34.layer3
        self.encoder4 = resnet34.layer4

        self.bridge = nn.Sequential(
            nn.Conv2d(filters[3], filters[3]*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, )
        )

        self.decoder1 = DecoderBlock(conv_in_channels=filters[3]*2, conv_out_channels=filters[3])
        self.decoder2 = DecoderBlock(conv_in_channels=filters[3], conv_out_channels=filters[2])
        self.decoder3 = DecoderBlock(conv_in_channels=filters[2], conv_out_channels=filters[1])
        self.decoder4 = DecoderBlock(conv_in_channels=filters[1], conv_out_channels=filters[0])
        self.decoder5 = DecoderBlock(
            conv_in_channels=filters[1], conv_out_channels=filters[0], up_in_channels=filters[0], up_out_channels=filters[0]
        )

        self.lastlayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters[0], out_channels=filters[0], kernel_size=2, stride=2),
            nn.Conv2d(filters[0], num_classes, kernel_size=3, padding=1, bias=False)
        )
    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x = self.pad(x)
        e1 = self.firstlayer(x)
        maxe1 = self.maxpool(e1)
        e2 = self.encoder1(maxe1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)
        
        c = self.bridge(e5)

        d1 = self.decoder1(c, e5)
        d2 = self.decoder2(d1, e4)
        d3 = self.decoder3(d2, e3)
        d4 = self.decoder4(d3, e2)
        d5 = self.decoder5(d4, e1)

        out = self.lastlayer(d5)


        if x.shape[-1] > self.input_size:
            crop_start = (x.shape[-1] - self.input_size) // 2
            crop_end = crop_start + self.input_size
            out = out[:, :, crop_start:crop_end, crop_start:crop_end]

        return out