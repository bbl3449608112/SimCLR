from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from augmentations import get_eval_transform, get_simclr_transform


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str = "data"
    image_size: int = 32
    train_limit: Optional[int] = 2000
    test_limit: Optional[int] = 1000
    seed: int = 42


class TwoViewDataset(Dataset):
    """Wrap a labeled image dataset and return two independent augmented views.

    The label is intentionally dropped in __getitem__ because SimCLR pretraining
    must not use labels.
    """

    def __init__(self, base_dataset: Dataset, transform: Callable):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.base_dataset[index]
        view_1 = self.transform(image)
        view_2 = self.transform(image)
        return view_1, view_2


def _limit_dataset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def build_pretrain_dataset(config: DatasetConfig) -> Dataset:
    base = CIFAR10(root=config.data_dir, train=True, download=True, transform=None)
    base = _limit_dataset(base, config.train_limit, config.seed)
    return TwoViewDataset(base, get_simclr_transform(config.image_size))


def build_linear_probe_datasets(config: DatasetConfig) -> Tuple[Dataset, Dataset]:
    transform = get_eval_transform(config.image_size)
    train = CIFAR10(root=config.data_dir, train=True, download=True, transform=transform)
    test = CIFAR10(root=config.data_dir, train=False, download=True, transform=transform)
    train = _limit_dataset(train, config.train_limit, config.seed)
    test = _limit_dataset(test, config.test_limit, config.seed + 1)
    return train, test


def build_prediction_dataset(config: DatasetConfig) -> Dataset:
    transform = get_eval_transform(config.image_size)
    test = CIFAR10(root=config.data_dir, train=False, download=True, transform=transform)
    return _limit_dataset(test, config.test_limit, config.seed + 1)
