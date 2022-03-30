import torch
from torch import nn


class PricePredictor(nn.Module):

    channels = [3, 16, 32, 32]
    linear_dims = [67_712, 1000, 7, 1]

    def __init__(self):
        super(PricePredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            *[self.conv_block(idx) for idx in range(len(self.channels) - 1)]
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_layers = nn.Sequential(
            *[self.linear_block(idx) for idx in range(len(self.linear_dims) - 1)]
        )
        self.prediction_layer = nn.Sequential(  # This part is where the weight value will be inputted
            nn.Linear(2, 1),
            nn.ReLU(inplace=False)
        )

        self.double()

    # Pass 1:  3x400x400 -> 16x396x396 -> 16x198x198
    # Pass 2: 16x198x198 -> 32x194x194 -> 32x 97x 97
    # Pass 3: 32x 97x 97 -> 32x 93x 93 -> 32x 46x 46
    def conv_block(self, in_chan_idx: int) -> nn.Module:
        conv = nn.Conv2d(in_channels=self.channels[in_chan_idx],
                         out_channels=self.channels[in_chan_idx + 1],
                         kernel_size=(5, 5),
                         stride=(1, 1))

        lower = - (6. / 5.) ** 0.5
        upper = (6. / 5.) ** 0.5
        nn.init.uniform_(conv.weight, lower, upper)

        return nn.Sequential(
            conv,
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )

    def linear_block(self, in_layer_idx: int) -> nn.Module:
        lin = nn.Linear(self.linear_dims[in_layer_idx], self.linear_dims[in_layer_idx + 1])

        lower = - (6. / float(self.linear_dims[in_layer_idx])) ** 0.5
        upper = (6. / float(self.linear_dims[in_layer_idx + 1])) ** 0.5
        nn.init.uniform_(lin.weight, lower, upper)

        return nn.Sequential(
            lin,
            nn.ReLU(inplace=False)
        )

    def forward(self, x_img: torch.Tensor, x_weight: torch.Tensor) -> torch.Tensor:
        x_img = self.conv_layers(x_img)
        x_img = self.flatten(x_img)
        x_img = self.linear_layers(x_img)
        return self.prediction_layer(torch.cat((x_img, x_weight), 1))
