import torch
from torch import nn


class PricePredictor(nn.Module):

    channels = [3, 16, 32]

    def __init__(self):
        super(PricePredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            *[self.conv_block(idx) for idx in range(len(self.channels))]
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(28_227, 1000),
            nn.ReLU(inplace=False),
            nn.Linear(1000, 1),
            nn.ReLU(inplace=False)
        )
        self.prediction_layer = nn.Sequential(  # input the weight value here
            nn.Linear(2, 1),
            nn.ReLU(inplace=False)
        )

    # Pass 1: 400x400x3  -> 396x396x16 -> 198x198x16
    # Pass 2: 198x198x16 -> 194x194x32 -> 97x97x32
    def conv_block(self, in_chan_idx: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=self.channels[in_chan_idx],
                      out_channels=self.channels[in_chan_idx + 1],
                      kernel_size=(5, 5),
                      stride=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )

    def forward(self, x_img: torch.Tensor, x_weight: torch.Tensor) -> torch.Tensor:
        x_img = self.conv_layers(x_img)
        x_img = self.flatten(x_img)
        x_img = self.linear_layers(x_img)
        return self.prediction_layer(torch.tensor([x_img.item(), x_weight.item()]))
