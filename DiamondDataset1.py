import os
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import random

import torch

from DiamondDataset import _DiamondDataset


class _DiamondDataset1(_DiamondDataset, ABC):

    def __init__(self):
        self._data_path = Path('Data') / 'Diamonds' / 'Diamonds'
        self._image_path = self._data_path / 'images'

        self.csv_data = pd.concat(
            [pd.read_csv(self._data_path / 'data_cushion.csv'), pd.read_csv(self._data_path / 'data_emerald.csv'),
             pd.read_csv(self._data_path / 'data_heart.csv'), pd.read_csv(self._data_path / 'data_oval.csv'),
             pd.read_csv(self._data_path / 'data_radiant.csv'), pd.read_csv(self._data_path / 'data_round.csv')],
            ignore_index=True)

        self._image_fnames = {
            'cushion': os.listdir(self._image_path / 'cushion'),
            'emerald': os.listdir(self._image_path / 'emerald'),
            'heart': os.listdir(self._image_path / 'heart'),
            'oval': os.listdir(self._image_path / 'oval'),
            'radiant': os.listdir(self._image_path / 'radiant'),
            'round': os.listdir(self._image_path / 'round')
        }

        self._partition_and_shuffle()
        self._remove_invalid_image_entries()

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self._random_squares(image)
        image = self._add_border(image)
        return image


class DiamondDataset1Train(_DiamondDataset1):

    def _partition_and_shuffle(self):
        self.csv_data = self.csv_data[self.csv_data['Use'] == 'Train']  # Remove test data
        self.csv_data.drop(columns='Data Url', inplace=True)  # Drop the url column
        self.csv_data = self.csv_data.sample(frac=1, ignore_index=True)  # Shuffle the data


class DiamondDataset1Test(_DiamondDataset1):

    def _partition_and_shuffle(self):
        self.csv_data = self.csv_data[self.csv_data['Use'] == 'Test']  # Remove train data
        self.csv_data.drop(columns='Data Url', inplace=True)  # Drop the url column
        self.csv_data = self.csv_data.sample(frac=1, ignore_index=True)  # Shuffle the data
