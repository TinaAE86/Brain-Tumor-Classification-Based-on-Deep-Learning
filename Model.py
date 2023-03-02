import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, X):
        Y = self.conv(X)
        return F.relu(Y)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, X):
        Y = self.conv(X)
        return F.relu(Y)


class ResBlk(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)  # 下采样
        self.conv2 = DoubleConv(in_ch, out_ch).conv  # 双层卷积
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)  # 直达通道

    def forward(self, X):
        X = self.conv1(X)
        Y = self.conv2(X)
        Y += self.conv3(X)
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ls_ch = [16, 32, 64, 128, 256]
        self.b1 = InConv(1, self.ls_ch[0])
        self.b2 = ResBlk(self.ls_ch[0], self.ls_ch[1])
        self.b3 = ResBlk(self.ls_ch[1], self.ls_ch[2])
        self.b4 = ResBlk(self.ls_ch[2], self.ls_ch[3])
        self.b5 = ResBlk(self.ls_ch[3], self.ls_ch[4])
        self.net = nn.Sequential(
            self.b1, self.b2, self.b3, self.b4, self.b5,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(self.ls_ch[4], 1),
            nn.Sigmoid()  # 一定要在输出加上Sigmoid()归一化，然后才可以输入给nn.BCELoss()计算loss
        )

    def forward(self, X):
        y = self.net(X)
        y = y.squeeze(-1)  # 输出预测标签时降维，避免训练时报错
        return y


if __name__ == '__main__':
    X = torch.rand(size=(1, 1, 240, 240))
    rn = ResNet()
    for layer in rn.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
