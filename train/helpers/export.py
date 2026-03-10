from pathlib import Path

import torch

from train.helpers.checkpoint import load_checkpoint_for_eval
from train.helpers.config import BEST_MODEL_PATH, IMAGE_SIZE, NUM_CHANNELS


ONNX_EXPORT_PATH = BEST_MODEL_PATH.parent / "spoofformer_best.onnx"


def export_best_model_to_onnx(
    checkpoint_path: Path = BEST_MODEL_PATH,
    onnx_path: Path = ONNX_EXPORT_PATH,
    opset_version: int = 17,
):
    model, checkpoint = load_checkpoint_for_eval(checkpoint_path)

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    return {
        "onnx_path": str(onnx_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
    }