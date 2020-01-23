import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        self.layer1 = nn.Conv2d(16, 8, kernel_size=2)
        self.layer2 = nn.Conv2d(8, 6, kernel_size=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = x.view(x.shape[0], 6, -1)

        return out
