import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class CnnV1(nn.Module):

    def __init__(self, depth=2, in_channels=1, out_channels=8, kernel_dim=3, mlp_dim=16, padding=1,
                 stride=1, max_pool=3, dropout=0.2, *args, **kwargs):
        super(CnnV1, self).__init__()
        # TODO: Save all params as attributes
        # TODO: Initialize weights
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_dim, padding=padding, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(out_channels, out_channels * 2, kernel_dim, padding=padding, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),

            # TODO: Derive a formula for channel dims
            # TODO: Create a depth loop for creating these dims
            # TODO: Create a separate conv block module
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_dim, padding=padding, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(out_channels * 4, out_channels * 8, kernel_dim, padding=padding, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),

            # TODO: Create a separate mlp module
            nn.Flatten(),
            nn.Linear(out_channels * 8, out_channels * 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 16, out_channels * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 8, mlp_dim),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
