'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as func

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'hg']


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, up1.shape[2:])
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, block, num_stacks=2, num_blocks=1, num_classes=16):

        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv3d(3, 1, kernel_size=1)

        # build hourglass modules
        ch1 = self.num_feats*block.expansion
        hg1, res1, fc1, score1, fc_1, score_1 = [], [], [], [], [], []
        for i in range(num_stacks):
            hg1.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res1.append(self._make_residual(block, self.num_feats, num_blocks))
            fc1.append(self._make_fc(ch1, ch1))
            score1.append(nn.Conv2d(ch1, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_1.append(nn.Conv2d(ch1, ch1, kernel_size=1, bias=True))
                score_1.append(nn.Conv2d(num_classes, ch1, kernel_size=1, bias=True))

        ch2 = self.num_feats * block.expansion
        hg2, res2, fc2, score2, fc_2, score_2 = [], [], [], [], [], []
        for i in range(num_stacks):
            hg2.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res2.append(self._make_residual(block, self.num_feats, num_blocks))
            fc2.append(self._make_fc(ch2, ch2))
            score2.append(nn.Conv2d(ch2, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_2.append(nn.Conv2d(ch2, ch2, kernel_size=1, bias=True))
                score_2.append(nn.Conv2d(num_classes, ch2, kernel_size=1, bias=True))

        ch3 = self.num_feats * block.expansion
        hg3, res3, fc3, score3, fc_3, score_3 = [], [], [], [], [], []
        for i in range(num_stacks):
            hg3.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res3.append(self._make_residual(block, self.num_feats, num_blocks))
            fc3.append(self._make_fc(ch3, ch3))
            score3.append(nn.Conv2d(ch3, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_3.append(nn.Conv2d(ch3, ch3, kernel_size=1, bias=True))
                score_3.append(nn.Conv2d(num_classes, ch3, kernel_size=1, bias=True))

        self.hg1 = nn.ModuleList(hg1)
        self.res1 = nn.ModuleList(res1)
        self.fc1 = nn.ModuleList(fc1)
        self.score1 = nn.ModuleList(score1)
        self.fc_1 = nn.ModuleList(fc_1)
        self.score_1 = nn.ModuleList(score_1)

        self.hg2 = nn.ModuleList(hg2)
        self.res2 = nn.ModuleList(res2)
        self.fc2 = nn.ModuleList(fc2)
        self.score2 = nn.ModuleList(score2)
        self.fc_2 = nn.ModuleList(fc_2)
        self.score_2 = nn.ModuleList(score_2)

        self.hg3 = nn.ModuleList(hg3)
        self.res3 = nn.ModuleList(res3)
        self.fc3 = nn.ModuleList(fc3)
        self.score3 = nn.ModuleList(score3)
        self.fc_3 = nn.ModuleList(fc_3)
        self.score_3 = nn.ModuleList(score_3)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        shape = x.shape[2:]

        out1 = []
        out2 = []
        out3 = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y1 = self.hg1[i](x)
            y1 = self.res1[i](y1)
            y1 = self.fc1[i](y1)
            score1 = self.score1[i](y1)
            end_score1 = F.interpolate(score1, shape)
            out1.append(end_score1)

            y2 = self.hg2[i](x)
            y2 = self.res2[i](y2)
            y2 = self.fc2[i](y2)
            score2 = self.score2[i](y2)
            end_score2 = F.interpolate(score2, shape)
            out2.append(end_score2)

            y3 = self.hg3[i](x)
            y3 = self.res3[i](y3)
            y3 = self.fc3[i](y3)
            score3 = self.score3[i](y3)
            end_score3 = F.interpolate(score3, shape)
            out3.append(end_score3)

            if i < self.num_stacks-1:
                fc_1 = self.fc_1[i](y1)
                score_1 = self.score_1[i](score1)
                x1 = x + fc_1 + score_1
                fc_2 = self.fc_2[i](y2)
                score_2 = self.score_2[i](score2)
                x2 = x + fc_2 + score_2
                fc_3 = self.fc_3[i](y3)
                score_3 = self.score_3[i](score3)
                x3 = x + fc_3 + score_3
                x = torch.stack((x1, x2, x3), dim=1)
                x = self.conv2(x)[:, 0]

        out1 = torch.cat(out1)
        out2 = torch.cat(out2)
        out3 = torch.cat(out3)

        return out1, out2, out3


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
