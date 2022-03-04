import os
from pathlib import Path
import pandas as pd
from abc import ABC

import torch

from DiamondDataset import _DiamondDataset


class _DiamondDataset2(_DiamondDataset, ABC):

    def __init__(self):
        self._data_path = Path('Data') / 'Diamonds2'
        self._image_path = self._data_path / 'images'

        self.csv_data = pd.concat(
            [pd.read_csv(self._data_path / 'data_cushion.csv'), pd.read_csv(self._data_path / 'data_emerald.csv'),
             pd.read_csv(self._data_path / 'data_heart.csv'), pd.read_csv(self._data_path / 'data_marquise.csv'),
             pd.read_csv(self._data_path / 'data_oval.csv'), pd.read_csv(self._data_path / 'data_pear.csv'),
             pd.read_csv(self._data_path / 'data_princess.csv'), pd.read_csv(self._data_path / 'data_round.csv')],
            ignore_index=True)

        self._image_fnames = {
            'cushion': os.listdir(self._image_path / 'cushion'),
            'emerald': os.listdir(self._image_path / 'emerald'),
            'heart': os.listdir(self._image_path / 'heart'),
            'marquise': os.listdir(self._image_path / 'marquise'),
            'oval': os.listdir(self._image_path / 'oval'),
            'pear': os.listdir(self._image_path / 'pear'),
            'princess': os.listdir(self._image_path / 'princess'),
            'round': os.listdir(self._image_path / 'round')
        }

        self._partition_and_shuffle()
        self._remove_invalid_image_entries()


class DiamondDataset2Train(_DiamondDataset2):

    def _partition_and_shuffle(self):
        self.csv_data = self.csv_data[self.csv_data['Use'] == 'Train']  # Remove test data
        self.csv_data = self.csv_data.sample(frac=1, ignore_index=True)  # Shuffle the data


class DiamondDataset2Test(_DiamondDataset2):

    def _partition_and_shuffle(self):
        self.csv_data = self.csv_data[self.csv_data['Use'] == 'Test']  # Remove train data
        self.csv_data = self.csv_data.sample(frac=1, ignore_index=True)  # Shuffle the data
