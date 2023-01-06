import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, RandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)


def load_dataset(base_dir, lengths=None, batch_size=64, shuffle=True, num_workers=4, pin_memory=False, seed=-1):
    if lengths is None:
        lengths = [0.8, 0.2]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    generator = torch.Generator()
    if seed >= 0:
        generator = generator.manual_seed(seed)
    dataset = ImageFolderWithPaths(root=base_dir, transform=transform)
    data_splits = random_split(dataset, lengths, generator=generator)

    data_loaders = map(lambda split: DataLoader(
        split, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker
    ), data_splits)
    return data_loaders


def load_dataset_curriculum(base_dir, p, knowledge_hierarchy, lengths=None, batch_size=64, shuffle=True, num_workers=4,
                            pin_memory=False, curriculum_sample_size=None, seed=-1):
    if lengths is None:
        lengths = [0.8, 0.2]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    generator = torch.Generator()
    if seed >= 0:
        generator = generator.manual_seed(seed)
    dataset = ImageFolderWithPaths(root=base_dir, transform=transform)

    data_splits = random_split(dataset, lengths, generator=generator)

    new_weights = []
    indices = data_splits[0].indices
    for ind in indices:
        image, label, path = dataset[ind]
        name = Path(path).stem.split("_")
        key = name[0] + "_" + name[-1]
        num_at_same_level = sum(1 for v in knowledge_hierarchy.values() if v == knowledge_hierarchy[key])
        new_weights.append(p[knowledge_hierarchy[key]] / num_at_same_level)

    sample_size = curriculum_sample_size or len(data_splits[0])
    train_sampler = WeightedRandomSampler(new_weights, sample_size, replacement=True)
    val_sampler = RandomSampler(data_splits[1])  # Test is always sampled uniformly
    test_sampler = RandomSampler(data_splits[2])  # Test is always sampled uniformly
    samplers = [train_sampler, val_sampler, test_sampler]

    data_loaders = map(lambda split, sampler: DataLoader(
        split, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker, sampler=sampler
    ), data_splits, samplers)
    return data_loaders


def load_image_by_shape_difficulty(dataset, shape, difficulty):
    images = dataset.imgs
    candidates = [
        i for i, image in enumerate(images) if shape in image[0] and f'diff{difficulty}' in image[0]
    ]

    selected_candidate_idx = random.choice(candidates)
    selected_candidate = dataset[selected_candidate_idx]  # a tuple of shape (data, target, path)

    return selected_candidate

def load_dataset_iita(base_dir):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = ImageFolderWithPaths(root=base_dir, transform=transform)
    return DataLoader(dataset, batch_size=1)
