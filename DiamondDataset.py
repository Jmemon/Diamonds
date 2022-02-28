from typing import List, Tuple
from abc import ABC
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class _DiamondDataset(Dataset, ABC):

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

        labels = self.csv_data.iloc[item].tolist()
        image_path = self._data_path / 'images' / str(labels[1]).lower()

        try:
            image = Image.open(image_path / (str(labels[0]) + '.jpg')).convert('RGB')
        except FileNotFoundError:
            image = Image.open(image_path / (str(labels[0]) + '.png')).convert('RGB')

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        return image, labels

    def _remove_no_image_entries(self):
        to_tensor = transforms.ToTensor()
        prod_video = Image.open(Path('Data') / 'Diamonds2' / 'images' / 'cushion' / '1781169.png').convert('RGB')
        prod_video = to_tensor(prod_video)

        to_drop = list()
        for idx, row in self.csv_data.iterrows():
            image_list = self._image_fnames[row['Shape'].lower()]
            image_id = str(row['Id'])

            if image_id + '.jpg' not in image_list and image_id + '.png' not in image_list:
                to_drop.append(idx)
                continue

            if row.isna().any():
                to_drop.append(idx)
                continue

            try:
                image = Image.open(self._image_path / row['Shape'].lower() / (image_id + '.jpg')).convert('RGB')
            except FileNotFoundError:
                image = Image.open(self._image_path / row['Shape'].lower() / (image_id + '.png')).convert('RGB')

            image = to_tensor(image)
            if image.shape != prod_video.shape:
                continue

            if (image == prod_video).all():
                to_drop.append(idx)

        self.csv_data.drop(index=to_drop, inplace=True)
        self.csv_data.set_index(pd.Index(range(self.csv_data.shape[0])), inplace=True)