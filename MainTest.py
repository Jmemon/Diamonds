from DiamondDataset1 import DiamondDataset1Train, DiamondDataset1Test
from DiamondDataset2 import DiamondDataset2Train, DiamondDataset2Test


if __name__ == '__main__':
    train_data1 = DiamondDataset1Train()
    test_data1 = DiamondDataset1Test()
    print(train_data1)
    print(test_data1)

    train_data2 = DiamondDataset2Train()
    test_data2 = DiamondDataset2Test()
    print(train_data2)
    print(test_data2)
