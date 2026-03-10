from pathlib import Path
from torch.utils.data import DataLoader

from train.dataset import (
    build_train_dataset,
    build_test_dataset,
)
from train.helpers.config import (
    DATA_DIR,
    BATCH_SIZE,
)


def build_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
):
    train_dataset = build_train_dataset(data_dir=data_dir)
    test_dataset = build_test_dataset(data_dir=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader