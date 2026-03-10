import torch

from train.models import SpoofFormer
from train.helpers.trainer import get_device


def build_model():
    return SpoofFormer()


def load_checkpoint_for_eval(checkpoint_path):
    """
    Load best checkpoint and return model in eval mode.
    """
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint