from DiamondDataset import DiamondDataset1

import pandas as pd

if __name__ == '__main__':
    data = DiamondDataset1()
    print(data.csv_data)

    for n in range(data.csv_data.shape[0]):
        print([data.csv_data.at[n, col] for col in data.csv_data])

    print(data.csv_data.drop_duplicates(subset=['Id'], ignore_index=True))
