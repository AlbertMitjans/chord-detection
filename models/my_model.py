import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(16, 8, kernel_size=7)
        self.layer2 = nn.Conv2d(8, 6, kernel_size=7)
        self.layer3 = nn.Linear(196, 50)
        self.layer4 = nn.Linear(50, 25)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = x.view(x.shape[0], 6, -1)
        x = self.layer3(x)
        x = self.layer4(x)
        out = x

        return out
