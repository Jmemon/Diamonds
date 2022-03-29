from torch import nn


class PricePredictor(nn.Module):

    def __init__(self):
        super(PricePredictor, self).__init__()
        in_layer = nn.Conv2d(in_channels=(3, 400, 400),
                             out_channels=(8, 200, 200),
                             kernel_size=(5, 5),
                             stride=2)  # layer dimensions organized as (C, H, W)
        h1 = nn.Conv2d(in_channels=(8, 200, 200),
                       out_channels=(8, 100, 100),
                       kernel_size=(5, 5),
                       stride=2)
        h2 = nn.Conv2d(in_channels=(8, 100, 100),
                       out_channels=(16, 50, 50),
                       kernel_size=(3, 3))
        flat = nn.Flatten()
        linear1 = nn.Linear(40000, 10000)
        linear2 = nn.Linear(10000, 1000)
        out_layer = nn.Linear(1000, 1)

    def forward(self, x):
        pass
