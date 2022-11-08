import os
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def load_dataset(base_dir, lengths=[0.8, 0.2], batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    dataset = ImageFolder(root=base_dir)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return random_split(data_loader, lengths)
