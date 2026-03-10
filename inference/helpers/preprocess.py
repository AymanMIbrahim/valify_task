import io

import numpy as np
from PIL import Image
from torchvision import transforms

from inference.helpers.config import IMAGE_SIZE


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image_from_path(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    transform = get_inference_transform()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    array = tensor.numpy().astype(np.float32)
    return array