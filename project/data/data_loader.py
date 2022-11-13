import os
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


def load_dataset(base_dir, lengths=None, batch_size=64, shuffle=True, num_workers=4, pin_memory=False):
    if lengths is None:
        lengths = [0.8, 0.2]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=base_dir, transform=transform)
    data_splits = random_split(dataset, lengths)
    data_loaders = map(lambda split: DataLoader(
        split, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    ), data_splits)
    return data_loaders
