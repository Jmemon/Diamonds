from DiamondDataset1 import DiamondDataset1Train, DiamondDataset1Test
from DiamondDataset2 import DiamondDataset2Train, DiamondDataset2Test


if __name__ == '__main__':
    train_data1 = DiamondDataset1Train()
    test_data1 = DiamondDataset1Test()

    print(train_data1)

    train_data2 = DiamondDataset2Train()
    test_data2 = DiamondDataset2Test()

    # /Diamonds2/images/cushion/1765490 is the only image that isn't 300x300
    # TODO: make sure we've filtered out 3d product video images

    """
    Dataset problems:
    - After filtering out everything we don't want, there isn't a ton of data
    - Every image is taken from an almost identical angle and distance, so this model will require images like the ones
    in this dataset to make accurate predictions
        - For an average user, this is a completely unrealistic assumption
        - Might even warrant searching for another dataset to supplement
    """
