from __future__ import annotations

from typing import Tuple

from torchvision import transforms


CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def get_simclr_transform(image_size: int = 32) -> transforms.Compose:
    """Return the stochastic augmentation pipeline used to create SimCLR views."""
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_eval_transform(image_size: int = 32) -> transforms.Compose:
    """Return deterministic preprocessing for linear probe and prediction display."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_display_transform(image_size: int = 32) -> transforms.Compose:
    """Return PIL-level preprocessing for visual prediction grids."""
    return transforms.Compose([transforms.Resize((image_size, image_size))])
