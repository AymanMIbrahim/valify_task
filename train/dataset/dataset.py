from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from train.helpers.config import DATA_DIR, IMAGE_SIZE, CLASS_TO_IDX


def get_default_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_default_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def read_split_file(split_path: Path) -> List[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    lines = split_path.read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in lines if line.strip()]
    return names


def build_samples_from_split_files(data_dir: Path = DATA_DIR) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Returns:
        train_samples: List[(image_path, label)]
        test_samples:  List[(image_path, label)]
    """

    live_train = read_split_file(data_dir / "LIVE_TRAIN.txt")
    live_test = read_split_file(data_dir / "LIVE_TEST.txt")
    spoof_train = read_split_file(data_dir / "SPOOF_TRAIN.txt")
    spoof_test = read_split_file(data_dir / "SPOOF_TEST.txt")

    train_samples = []
    test_samples = []

    for image_name in live_train:
        image_path = data_dir / image_name
        if image_path.exists():
            train_samples.append((image_path, CLASS_TO_IDX["live"]))
        else:
            print(f"[WARNING] Missing train live image: {image_path}")

    for image_name in spoof_train:
        image_path = data_dir / image_name
        if image_path.exists():
            train_samples.append((image_path, CLASS_TO_IDX["spoof"]))
        else:
            print(f"[WARNING] Missing train spoof image: {image_path}")

    for image_name in live_test:
        image_path = data_dir / image_name
        if image_path.exists():
            test_samples.append((image_path, CLASS_TO_IDX["live"]))
        else:
            print(f"[WARNING] Missing test live image: {image_path}")

    for image_name in spoof_test:
        image_path = data_dir / image_name
        if image_path.exists():
            test_samples.append((image_path, CLASS_TO_IDX["spoof"]))
        else:
            print(f"[WARNING] Missing test spoof image: {image_path}")

    return train_samples, test_samples


class FaceAntiSpoofDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform if transform is not None else get_default_train_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_train_dataset(data_dir: Path = DATA_DIR, transform=None) -> FaceAntiSpoofDataset:
    train_samples, _ = build_samples_from_split_files(data_dir)
    return FaceAntiSpoofDataset(
        samples=train_samples,
        transform=transform if transform is not None else get_default_train_transform(),
    )


def build_test_dataset(data_dir: Path = DATA_DIR, transform=None) -> FaceAntiSpoofDataset:
    _, test_samples = build_samples_from_split_files(data_dir)
    return FaceAntiSpoofDataset(
        samples=test_samples,
        transform=transform if transform is not None else get_default_eval_transform(),
    )