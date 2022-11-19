import torch.nn
import torch.nn as nn


class CnnV1(nn.Module):

    def __init__(self, depth=2, in_channels=1, out_channels=8, kernel_dim=3, mlp_dim=3, padding=1,
                 stride=1, max_pool=3, dropout=0.2, batch_size=32, img_channels=1, img_dims=(96, 96), *args, **kwargs):
        super(CnnV1, self).__init__()
        # TODO: Save all params as attributes
        # TODO: Initialize weights
        # TODO: Create a depth loop for creating these dims
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_dim=kernel_dim, padding=padding,
                      stride=stride, max_pool=max_pool, filter_scaling=2),
            ConvBlock(in_channels=out_channels * 2, out_channels=out_channels * 4, kernel_dim=kernel_dim,
                      padding=padding, stride=stride, max_pool=max_pool, filter_scaling=2),
            nn.Flatten(),
        )
        encoder_channels = self.conv_blocks(torch.empty(batch_size, img_channels, *img_dims)).size(-1)
        self.linear = FCC(in_channels=encoder_channels, out_channels=out_channels, mlp_dim=mlp_dim, scaling_factor=16,
                          dropout=dropout)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels=1, out_channels=8, kernel_dim=3, padding=1,
                 stride=1, max_pool=3, filter_scaling=2, *args, **kwargs):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            # Formula for conv dims [(W−K+2P)/S]+1
            nn.Conv2d(in_channels, out_channels, kernel_dim, padding=padding, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(out_channels, out_channels * filter_scaling, kernel_dim, padding=padding, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(max_pool)
        )

    def forward(self, x):
        return self.layers(x)


class FCC(nn.Module):

    def __init__(self, in_channels=64, out_channels=1024, mlp_dim=3, scaling_factor=16, dropout=0.2):
        super(FCC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels * scaling_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * scaling_factor, out_channels * scaling_factor // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * scaling_factor // 2, mlp_dim),
        )

    def forward(self, x):
        return self.layers(x)
