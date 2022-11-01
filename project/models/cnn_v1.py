import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class CnnV1(nn.Module):

    def __init__(self, ):
        super(CnnV1, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.Linear(64 * 3 * 3, 128),
            F.relu,
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            F.relu,
            nn.Dropout(0.2),
            nn.Linear(64, 16),
        ])

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
