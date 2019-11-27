import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from baselines.utils.consts import Split, CsvField


class ClassificationDataset(Dataset):
    """Image classification dataset."""

    def __init__(self, subset, csv_path, data_folder, size=(224, 224)):
        """
        Args:
            subset (string): Subset of the dataset to load (either "train", "validation" or "test").
                on a sample.
        """
        assert size[0] < 256 and size[1] < 256
        assert subset not in [attr for attr in dir(Split) if not attr.startswith('__')]
        self.csv_path = csv_path
        self.data_path = data_folder
        self.subset = subset
        self.size = size
        self.transform = transforms.Compose(self._get_transforms_list())
        self.labels2idx, self.idx2labels, self.metadata = self._get_metadata()

    def _get_transforms_list(self):
        if self.subset == Split.TRAIN:
            return_transform = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            return_transform = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        return return_transform

    def _get_metadata(self):
        df = pd.read_csv(self.csv_path)
        labels2idx = {}
        idx2labels = {}
        for idx, label in enumerate(df["Label"].unique()):
            labels2idx[label] = idx
            idx2labels[idx] = label
        metadata = df[df['Split'] == self.subset]
        return labels2idx, idx2labels, metadata

    def get_idx2labels(self):
        return self.idx2labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path,
                                self.metadata.iloc[idx, CsvField.IMAGE_FILE])
        image = Image.open(img_name).convert('RGB')

        label = self.metadata.iloc[idx, CsvField.TARGET]
        label_idx = self.labels2idx[label]

        image = self.transform(image)

        return image, label_idx


class RegressionDataset(Dataset):
    """Image regression dataset."""

    def __init__(self, subset, csv_path, data_folder, size=(224, 224)):
        """
        Args:
            subset (string): Subset of the dataset to load (either "train", "validation" or "test").
                on a sample.
        """
        assert subset not in [attr for attr in dir(Split) if not attr.startswith('__')]
        self.csv_path = csv_path
        self.data_path = data_folder
        self.subset = subset
        self.size = size
        self.transform = transforms.Compose(self._get_transforms_list())
        self.metadata = self._get_metadata()

    def _get_transforms_list(self):
        if self.subset == Split.TRAIN:
            return_transform = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            return_transform = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        return return_transform

    def _get_metadata(self):
        df = pd.read_csv(self.csv_path)
        metadata = df[df['Split'] == self.subset]
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path,
                                self.metadata.iloc[idx, CsvField.IMAGE_FILE])
        image = Image.open(img_name).convert('RGB')

        label = self.metadata.iloc[idx, CsvField.TARGET]
        target_value = torch.tensor([label], dtype=torch.float)
        image = self.transform(image)

        return image, target_value
