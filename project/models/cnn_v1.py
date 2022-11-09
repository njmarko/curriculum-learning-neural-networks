import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class CnnV1(nn.Module):

    def __init__(self, depth=2, in_channels=1, out_channels=16, kernel_dim=3, mlp_dim=16, dropout=0.2):
        super(CnnV1, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # nn.Conv2d(32, 64, 3),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, 3),
            # nn.ReLU(),
            # nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
