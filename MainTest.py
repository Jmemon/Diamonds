from DiamondDataset1 import DiamondDataset1Train, DiamondDataset1Test
from DiamondDataset2 import DiamondDataset2Train, DiamondDataset2Test

import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data1 = DiamondDataset1Train()
    #test_data1 = DiamondDataset1Test()

    #train_data2 = DiamondDataset2Train()
    #test_data2 = DiamondDataset2Test()

    #plt.figure()

    #for i in range(9):
    #    plt.subplot(3, 3, i + 1)
    #    plt.imshow(train_data2[i][0].permute(1, 2, 0).numpy())

    #plt.show()

    train_data1[3]

    """
    Dataset problems:
    - After filtering out everything we don't want, there isn't a ton of data
    - Every image is taken from an almost identical angle and distance, so this model will require images like the ones
    in this dataset to make accurate predictions
        - Will not be robust
        - For an average user, this is a completely unrealistic assumption
        - Might even warrant searching for another dataset to supplement
        - Other options:
            - Add borders to images with random stuff in it
            - Rotate images, then crop/resize/border to make it's size compatible with other samples
            - Change some pixels to random values
    """
