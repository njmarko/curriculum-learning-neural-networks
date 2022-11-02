import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class CnnV1(nn.Module):

    def __init__(self, ):
        super(CnnV1, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
        ])

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
