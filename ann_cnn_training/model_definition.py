import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_CNN(nn.Module):
    def __init__(self, idim, odim, hdim, stencil, dropout=0):
        super().__init__()

        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.dropout_prob = dropout
        self.stencil = stencil
        self.fac = np.floor(0.5 * self.stencil)

        # assume normalized data as input
        # same activation for all layers
        # same dropout probabilities for all layers

        # Applying multiple 3x3 conv layers than just one stencilxstencil layer performs better
        if self.fac == 1:
            self.conv1 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0
            )
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5 * self.dropout_prob)
            # print('CNN 1')
        elif self.fac == 2:
            self.conv1 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0
            )
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5 * self.dropout_prob)
            # print('-CNN 1')
            self.conv2 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0
            )
            self.act_cnn2 = nn.ReLU()
            self.dropout0_2 = nn.Dropout(p=0.5 * self.dropout_prob)
            # print('-CNN 2')
        elif self.fac == 3:  # this is not ready yet, and might not be needed for my study
            self.conv1 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=self.stencil, stride=1, padding=0
            )
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5 * self.dropout_prob)

        # can define a block and divide it into blocks as well
        self.layer1 = nn.Linear(idim, hdim)  # ,dtype=torch.float16)
        self.act1 = (
            nn.LeakyReLU()
        )  # nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm1 = nn.BatchNorm1d(hdim)

        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.layer2 = nn.Linear(hdim, hdim)
        self.act2 = (
            nn.LeakyReLU()
        )  # nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm2 = nn.BatchNorm1d(hdim)
        # -------------------------------------------------------
        self.layer3 = nn.Linear(hdim, hdim)
        self.act3 = (
            nn.LeakyReLU()
        )  # nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm3 = nn.BatchNorm1d(hdim)
        # -------------------------------------------------------
        self.layer4 = nn.Linear(hdim, hdim)
        self.act4 = (
            nn.LeakyReLU()
        )  # nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm4 = nn.BatchNorm1d(2 * hdim)
        # --------------------------------------------------------
        self.layer5 = nn.Linear(hdim, hdim)
        self.act5 = (
            nn.LeakyReLU()
        )  # nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm5 = nn.BatchNorm1d(hdim)
        # -------------------------------------------------------
        self.layer6 = nn.Linear(hdim, 2 * odim)
        self.act6 = (
            nn.LeakyReLU()
        )  # nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm6 = nn.BatchNorm1d(2 * odim)

        self.output = nn.Linear(2 * odim, odim)

    def forward(self, x):
        if self.fac == 1:
            x = torch.squeeze(self.dropout0(self.act_cnn(self.conv1(x))))
        elif self.fac == 2:
            x = torch.squeeze(self.dropout0(self.act_cnn(self.conv1(x))))
            x = torch.squeeze(self.dropout0_2(self.act_cnn2(self.conv2(x))))

        # elif fac == 2:
        #    x = self.conv1(x)
        # elif fac == 3:

        # print(f'new shape: {x.shape}')
        x = self.dropout(self.act1(self.layer1(x)))
        x = self.dropout(self.act2(self.layer2(x)))
        x = self.dropout(self.act3(self.layer3(x)))
        x = self.dropout(self.act4(self.layer4(x)))
        x = self.dropout(self.act5(self.layer5(x)))
        x = self.dropout(self.act6(self.layer6(x)))
        x = self.output(x)

        # x = self.dropout1(self.bnorm1(self.act1(self.layer1(x))))
        # x = self.dropout2(self.bnorm2(self.act2(self.layer2(x))))
        # x = self.dropout3(self.bnorm3(self.act3(self.layer3(x))))
        # x = self.dropout4(self.bnorm4(self.act4(self.layer4(x))))
        # x = self.output(x)

        return x

    def gaussian_dropout(self, x):
        S = x.shape
        vec = torch.normal(mean=torch.ones(S), std=torch.ones(S))
        return x * vec

    # calculates total number of learnable parameters
    def totalparams(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement()

        return param_size

    # computes total model size in MBs
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


# ===================== ATTENTION UNET MODEL DEFINITION =====================================
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
        if self.dropout_prob > 0:
            x1 = self.dropout(x1)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        if self.dropout_prob > 0:
            x2 = self.dropout(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        if self.dropout_prob > 0:
            x3 = self.dropout(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        if self.dropout_prob > 0:
            x4 = self.dropout(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        if self.dropout_prob > 0:
            x5 = self.dropout(x5)

        # print(f'x5 {x5.shape}')
        g5 = self.up5(x5)
        # print(f'g5 {g5.shape}')
        x4 = self.attn5(g=g5, x=x4)
        # print(f'x4 {x4.shape}')
        g5 = torch.cat((g5, x4), dim=1)
        # print(f'g5_2 {g5.shape}')
        g5 = self.upconv5(g5)
        # print(f'g5_3 {g5.shape}')

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


class ANN_CNN10(nn.Module):
    def __init__(self, idim, odim, hdim, stencil, dropout=0):
        super().__init__()

        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.dropout_prob = dropout
        self.stencil = stencil
        self.fac = np.floor(0.5 * self.stencil)

        # assume normalized data as input
        # same activation for all layers
        # same dropout probabilities for all layers

        # Applying multiple 3x3 conv layers than just one stencilxstencil layer performs better
        if self.fac == 1:
            self.conv1 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0
            )
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5 * self.dropout_prob)

        elif self.fac == 2:
            self.conv1 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0
            )
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5 * self.dropout_prob)
            # print('-CNN 1')
            self.conv2 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0
            )
            self.act_cnn2 = nn.ReLU()
            self.dropout0_2 = nn.Dropout(p=0.5 * self.dropout_prob)

        elif self.fac == 3:  # this is not ready yet, and might not be needed for my study
            self.conv1 = nn.Conv2d(
                in_channels=idim, out_channels=idim, kernel_size=self.stencil, stride=1, padding=0
            )
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5 * self.dropout_prob)

        # can define a block and divide it into blocks as well
        self.layer1 = nn.Linear(idim, hdim)  # ,dtype=torch.float16)
        self.act1 = nn.LeakyReLU()
        # self.bnorm1 = nn.BatchNorm1d(hdim)

        self.dropout = nn.Dropout(p=self.dropout_prob)

        self.layer2 = nn.Linear(hdim, hdim)
        self.act2 = nn.LeakyReLU()
        # self.bnorm2 = nn.BatchNorm1d(hdim)
        # -------------------------------------------------------
        self.layer3 = nn.Linear(hdim, hdim)
        self.act3 = nn.LeakyReLU()
        # self.bnorm3 = nn.BatchNorm1d(hdim)
        # -------------------------------------------------------
        self.layer4 = nn.Linear(hdim, hdim)
        self.act4 = nn.LeakyReLU()
        # self.bnorm4 = nn.BatchNorm1d(2 * hdim)
        # --------------------------------------------------------
        self.layer5 = nn.Linear(hdim, hdim)
        self.act5 = nn.LeakyReLU()
        # self.bnorm5 = nn.BatchNorm1d(hdim)
        # -------------------------------------------------------

        self.layer6 = nn.Linear(hdim, hdim)
        self.act6 = nn.LeakyReLU()
        # -------------------------------------------------------
        self.layer7 = nn.Linear(hdim, hdim)
        self.act7 = nn.LeakyReLU()
        # -------------------------------------------------------
        self.layer8 = nn.Linear(hdim, hdim)
        self.act8 = nn.LeakyReLU()
        # -------------------------------------------------------
        self.layer9 = nn.Linear(hdim, hdim)
        self.act9 = nn.LeakyReLU()
        # -------------------------------------------------------
        self.layer10 = nn.Linear(hdim, 2 * odim)
        self.act10 = nn.LeakyReLU()
        # -------------------------------------------------------

        self.output = nn.Linear(2 * odim, odim)

    def forward(self, x):
        if self.fac == 1:
            x = torch.squeeze(self.dropout0(self.act_cnn(self.conv1(x))))
        elif self.fac == 2:
            x = torch.squeeze(self.dropout0(self.act_cnn(self.conv1(x))))
            x = torch.squeeze(self.dropout0_2(self.act_cnn2(self.conv2(x))))

        x = self.dropout(self.act1(self.layer1(x)))
        x = self.dropout(self.act2(self.layer2(x)))
        x = self.dropout(self.act3(self.layer3(x)))
        x = self.dropout(self.act4(self.layer4(x)))
        x = self.dropout(self.act5(self.layer5(x)))
        x = self.dropout(self.act6(self.layer6(x)))
        x = self.dropout(self.act7(self.layer7(x)))
        x = self.dropout(self.act8(self.layer8(x)))
        x = self.dropout(self.act9(self.layer9(x)))
        x = self.dropout(self.act10(self.layer10(x)))
        x = self.output(x)

        return x

    # calculates total number of learnable parameters
    def totalparams(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement()

        return param_size

    # computes total model size in MBs
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
