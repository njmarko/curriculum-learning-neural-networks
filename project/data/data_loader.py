import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset(base_dir, lengths=None, batch_size=64, shuffle=True, num_workers=4, pin_memory=False, seed=None):
    if lengths is None:
        lengths = [0.8, 0.2]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    generator = torch.Generator()
    if seed:
        generator = generator.manual_seed(seed)
    dataset = ImageFolder(root=base_dir, transform=transform)
    data_splits = random_split(dataset, lengths, generator=generator)

    data_loaders = map(lambda split: DataLoader(
        split, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker
    ), data_splits)
    return data_loaders
