# Model definition: init and forward function for attention unet only
# No dropout has been used for Attention U-Net (Oktay et al. (2018))

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=ch_out,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_x, F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.Wx = nn.Sequential(
            nn.Conv2d(
                in_channels=F_x,
                out_channels=F_int,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(F_int),
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(
                in_channels=F_g,
                out_channels=F_int,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(F_int),
        )

        self.Psi = nn.Sequential(
            nn.Conv2d(
                in_channels=F_int,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        x1 = self.Wx(x)
        g1 = self.Wg(g)
        alp = self.Psi(self.relu(x1 + g1))
        x = x * alp  # scaling skip connection by attention coefficients
        return x


class Attention_UNet(nn.Module):
    def __init__(self, ch_in, ch_out, dropout=0.2):
        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dropout_prob = dropout

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(
            p=self.dropout_prob, inplace=False
        )  # applying dropout only during downsampling

        self.conv1 = Conv_block(ch_in=self.ch_in, ch_out=64)
        self.conv2 = Conv_block(ch_in=64, ch_out=128)
        self.conv3 = Conv_block(ch_in=128, ch_out=256)
        self.conv4 = Conv_block(ch_in=256, ch_out=512)
        self.conv5 = Conv_block(ch_in=512, ch_out=1024)

        self.up5 = Upsample(ch_in=1024, ch_out=512)
        self.attn5 = Attention_block(F_x=512, F_g=512, F_int=256)
        self.upconv5 = Conv_block(ch_in=1024, ch_out=512)

        self.up4 = Upsample(ch_in=512, ch_out=256)
        self.attn4 = Attention_block(F_x=256, F_g=256, F_int=128)
        self.upconv4 = Conv_block(ch_in=512, ch_out=256)

        self.up3 = Upsample(ch_in=256, ch_out=128)
        self.attn3 = Attention_block(F_x=128, F_g=128, F_int=64)
        self.upconv3 = Conv_block(ch_in=256, ch_out=128)

        self.up2 = Upsample(ch_in=128, ch_out=64)
        self.attn2 = Attention_block(F_x=64, F_g=64, F_int=32)
        self.upconv2 = Conv_block(ch_in=128, ch_out=64)

        self.conv1x1 = nn.Conv2d(
            in_channels=64, out_channels=self.ch_out, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1d = self.dropout(x1)

        x2 = self.maxpool(x1d)
        x2 = self.conv2(x2)
        x2d = self.dropout(x2)

        x3 = self.maxpool(x2d)
        x3 = self.conv3(x3)
        x3d = self.dropout(x3)

        x4 = self.maxpool(x3d)
        x4 = self.conv4(x4)
        x4d = self.dropout(x4)

        x5 = self.maxpool(x4d)
        x5 = self.conv5(x5)
        x5d = self.dropout(x5)

        g5 = self.up5(x5d)
        x4 = self.attn5(g=g5, x=x4)
        g5 = torch.cat((g5, x4), dim=1)
        g5 = self.upconv5(g5)

        g4 = self.up4(g5)
        x3 = self.attn4(g=g4, x=x3)
        g4 = torch.cat((g4, x3), dim=1)
        g4 = self.upconv4(g4)

        g3 = self.up3(g4)
        x2 = self.attn3(g=g3, x=x2)
        g3 = torch.cat((g3, x2), dim=1)
        g3 = self.upconv3(g3)

        g2 = self.up2(g3)
        x1 = self.attn2(g=g2, x=x1)
        g2 = torch.cat((g2, x1), dim=1)
        g2 = self.upconv2(g2)

        # modifying to output dimensions
        x = self.conv1x1(g2)

        return x

    def totalparams(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement()

        return param_size

    def totalsize(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        # print('model size: {:.3f}MB'.format(size_all_mb))

        return size_all_mb
