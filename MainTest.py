from DiamondDataset1 import DiamondDataset1Train, DiamondDataset1Test
from DiamondDataset2 import DiamondDataset2Train, DiamondDataset2Test

import torch

if __name__ == '__main__':
    train_data1 = DiamondDataset1Train()
    test_data1 = DiamondDataset1Test()

    train_data2 = DiamondDataset2Train()
    test_data2 = DiamondDataset2Test()

    # /Diamonds2/images/cushion/1765490 is the only image that isn't 300x300
    # TODO: filter out 3d product video images
