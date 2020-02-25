import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(16, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x[0])
        out2 = self.conv2(x[1])
        out3 = self.conv2(x[2])

        return out1, out2, out3
