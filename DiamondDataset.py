from typing import List, Tuple
from abc import ABC, abstractmethod
import pandas as pd
from PIL import Image
from pathlib import Path
import random

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
        image = self._preprocess_image(image)
        return image, labels

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self._random_squares(image)
        image = self._add_border(image)
        return image

    def _add_border(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds an extra 50 pixels above, below, left, and right of image.
        :return: torch.Tensor modified to include more images
        """
        assert image.shape == (3, 300, 300), f'_add_border() expected a 3x300x300 image tensor, got {image.shape}'

        # TODO: Randomly bias the random value generation toward certain channels for increased variability of examples?
        #channel = random.choice([0, 1, 2])  # r, g, b

        new_image = torch.rand((image.shape[0], image.shape[1] + 100, image.shape[2] + 100))
        new_image[:, 50: 350, 50: 350] = image

        return new_image

    def _random_squares(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds up to 10 random 10x10 squares in random locations of the main image
        :param image: torch.Tensor to modify
        :return: torch.Tensor that is modified version of original
        """
        assert image.shape == (3, 300, 300), f'Expected an unaltered image, got an image of shape {image.shape}'

        num_squares = random.randint(1, 10)  # integer in [1, 5]

        for _ in range(num_squares):
            start_row = random.randint(0, 290)
            start_col = random.randint(0, 290)
            image[:, start_row: start_row + 10, start_col: start_col + 10] = torch.rand((3, 10, 10))

        return image

    @abstractmethod
    def _partition_and_shuffle(self):
        pass

    def _remove_invalid_image_entries(self):
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
                to_drop.append(idx)
                continue

            if (image == prod_video).all():
                to_drop.append(idx)

        self.csv_data.drop(index=to_drop, inplace=True)
        self.csv_data.set_index(pd.Index(range(self.csv_data.shape[0])), inplace=True)