import os
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import pandas as pd
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class _DiamondDataset2(Dataset, ABC):

    def __init__(self):
        self._data_path = Path('Data') / 'Diamonds2'

        self.csv_data = pd.concat(
            [pd.read_csv(self._data_path / 'data_cushion.csv'), pd.read_csv(self._data_path / 'data_emerald.csv'),
             pd.read_csv(self._data_path / 'data_heart.csv'), pd.read_csv(self._data_path / 'data_marquise.csv'),
             pd.read_csv(self._data_path / 'data_oval.csv'), pd.read_csv(self._data_path / 'data_pear.csv'),
             pd.read_csv(self._data_path / 'data_princess.csv'), pd.read_csv(self._data_path / 'data_round.csv')],
            ignore_index=True)

        self._clean_and_shuffle()
        self._remove_no_image_entries()

    def __str__(self):
        return str(self.csv_data)

    def __len__(self) -> int:
        return self.csv_data.shape[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Returns an item by index from the dataset.
        :param item: uint representing index
        :return: 2-tuple with the first entry being the labels associated with the image and the second being the image
        """
        if item >= self.csv_data.shape[0]:
            raise IndexError(f'Index {item} is too large. Max index is {self.csv_data.shape[0] - 1}.')

        if item < 0:
            raise IndexError(f'Index {item} is too small. Min index is 0.')

        labels = self.csv_data.iloc[[item]].values.tolist()[0]
        image_path = self._data_path / 'images' / labels[1].lower()

        try:
            image = Image.open(image_path / (str(labels[0]) + '.jpg')).convert('RGB')
        except FileNotFoundError:
            image = Image.open(image_path / (str(labels[0]) + '.png')).convert('RGB')

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        return image, labels

    @abstractmethod
    def _clean_and_shuffle(self):
        pass

    def _remove_no_image_entries(self):
        image_path = self._data_path / 'images'

        image_fnames = {
            'cushion': os.listdir(image_path / 'cushion'),
            'emerald': os.listdir(image_path / 'emerald'),
            'heart': os.listdir(image_path / 'heart'),
            'marquise': os.listdir(image_path / 'marquise'),
            'oval': os.listdir(image_path / 'oval'),
            'pear': os.listdir(image_path / 'pear'),
            'radiant': os.listdir(image_path / 'princess'),
            'round': os.listdir(image_path / 'round')
        }

        to_drop = list()
        for idx, row in self.csv_data.iterrows():
            image_list = image_fnames[row['Shape'].lower()]
            img_id = str(row['Id'])

            if img_id + '.jpg' not in image_list and img_id + '.png' not in image_list:
                to_drop.append(idx)

        self.csv_data.drop(index=to_drop, inplace=True)
        self.csv_data = self.csv_data.reindex(range(self.csv_data.shape[0]))


class DiamondDataset2Train(_DiamondDataset2):

    def _clean_and_shuffle(self):
        self.csv_data = self.csv_data[self.csv_data['Use'] == 'Train']  # Remove test data
        self.csv_data = self.csv_data.sample(frac=1, ignore_index=True)  # Shuffle the data


class DiamondDataset2Test(_DiamondDataset2):

    def _clean_and_shuffle(self):
        self.csv_data = self.csv_data[self.csv_data['Use'] == 'Train']  # Remove train data
        self.csv_data = self.csv_data.sample(frac=1, ignore_index=True)  # Shuffle the data