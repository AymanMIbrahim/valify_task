import torch

from inference.models import SpoofFormer
from inference.helpers.config import CHECKPOINT_PATH


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    return SpoofFormer()


def load_inference_model(checkpoint_path=CHECKPOINT_PATH):
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device, checkpoint