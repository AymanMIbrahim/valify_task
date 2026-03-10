import numpy as np

from inference.helpers.config import IDX_TO_CLASS
from inference.helpers.onnx_session import get_onnx_session
from inference.helpers.preprocess import (
    load_image_from_path,
    load_image_from_bytes,
    preprocess_pil_image,
)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def run_prediction(image_array: np.ndarray):
    session = get_onnx_session()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: image_array})[0]
    probs = softmax(logits)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_index": pred_idx,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": {
            "spoof": float(probs[0]),
            "live": float(probs[1]),
        },
    }


def predict_image_path(image_path: str):
    image = load_image_from_path(image_path)
    image_array = preprocess_pil_image(image)
    return run_prediction(image_array)


def predict_image_bytes(image_bytes: bytes):
    image = load_image_from_bytes(image_bytes)
    image_array = preprocess_pil_image(image)
    return run_prediction(image_array)