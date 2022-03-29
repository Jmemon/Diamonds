from DiamondDataset2 import DiamondDataset2

from torch.utils.data import random_split
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

if __name__ == '__main__':
    diamonds = DiamondDataset2()

    train_len = int(0.7 * len(diamonds))
    train, test = random_split(diamonds, (train_len, len(diamonds) - train_len))

    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=128, shuffle=False)

    print(train[0][0].shape)
    print(train[0][0])
    [print(tens) for tens in train[0][1]]

    #plt.figure()

    #for i in range(9):
    #    plt.subplot(3, 3, i + 1)
    #    plt.imshow(train_data2[i][0].permute(1, 2, 0).numpy())

    #plt.show()

    # TODO: Next steps
    #  After that try to build a model and throw some stuff in there
    #  Do price prediction first. Try two things: the image and weight, then just the labels. See how it does.
    #  Second give the weight and the image, then try to get the right labels

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
