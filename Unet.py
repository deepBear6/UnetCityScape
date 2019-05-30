import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class DownScale(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownScale, self).__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.down_block(x)
        return x



class UpScale(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpScale, self).__init__()

        # self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners=True)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2) 
        self.doubleconv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):

        x1 = self.upsample(x1, output_size=x2.shape)
        assert x1.shape == x2.shape, f"up size:{x1.shape}, previous:{x2.shape}"
        x = torch.cat([x2, x1], dim=1)
        x = self.doubleconv(x)
        return x



class Unet(nn.Module):

    def __init__(self, input_channels, num_classes):
        super(Unet, self).__init__()
        self.inconv = DoubleConv(input_channels, 64)
        self.down_block1 = DownScale(64, 128)
        self.down_block2 = DownScale(128, 256)
        
        self.down_block3 = DownScale(256, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.down_block4 = DownScale(512, 1024)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.up_block1 = UpScale(1024, 512)
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.up_block2 = UpScale(512, 256)
        self.dropout4 = nn.Dropout(p=0.5)
        
        self.up_block3 = UpScale(256, 128)
        self.up_block4 = UpScale(128, 64)
        self.outputmap = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        x4 = self.down_block3(x3)
        x4 = self.dropout1(x4)
        x5 = self.down_block4(x4)
        x5 = self.dropout2(x5)
        x = self.up_block1(x5, x4)
        x = self.dropout3(x)
        x = self.up_block2(x, x3)
        x = self.dropout4(x)
        x = self.up_block3(x, x2)
        x = self.up_block4(x, x1)
        x = self.outputmap(x)
        y = self.softmax(x)
        return y

