from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class DiamondDataset1(Dataset):

    def __init__(self):
        _data_path = Path('Data') / 'Diamonds' / 'Diamonds'

        self.csv_data = pd.concat(
            [pd.read_csv(_data_path / 'data_cushion.csv'), pd.read_csv(_data_path / 'data_emerald.csv'),
             pd.read_csv(_data_path / 'data_heart.csv'), pd.read_csv(_data_path / 'data_oval.csv'),
             pd.read_csv(_data_path / 'data_radiant.csv'), pd.read_csv(_data_path / 'data_round.csv')],
            ignore_index=True)

        self.image_data_paths = {
            'cushion': _data_path / 'images' / 'cushion',
            'emerald': _data_path / 'images' / 'emerald',
            'heart': _data_path / 'images' / 'heart',
            'oval': _data_path / 'images' / 'oval',
            'radiant': _data_path / 'images' / 'radiant',
            'round': _data_path / 'images' / 'round'
        }

    def __len__(self) -> int:
        return sum([val.shape[0] for val in self.csv_data.values()])

    def __getitem__(self, item: Tuple[str, int]) -> Tuple[List[str], torch.Tensor]:
        return self.csv_data[item[0]].iloc[[item[1]]].values.tolist()
