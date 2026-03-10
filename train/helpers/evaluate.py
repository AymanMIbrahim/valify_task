from typing import Dict, List

import torch

from train.helpers.checkpoint import load_checkpoint_for_eval
from train.helpers.dataloaders import build_dataloaders
from train.helpers.trainer import get_device
from train.helpers.config import BEST_MODEL_PATH


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    """
    Label convention:
    0 = spoof
    1 = live
    """

    tp = sum((p == 1 and y == 1) for y, p in zip(labels, preds))
    tn = sum((p == 0 and y == 0) for y, p in zip(labels, preds))
    fp = sum((p == 1 and y == 0) for y, p in zip(labels, preds))
    fn = sum((p == 0 and y == 1) for y, p in zip(labels, preds))

    total = len(labels)

    accuracy = (tp + tn) / total if total else 0

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    # Anti-spoofing metrics

    num_spoof = tn + fp
    num_live = tp + fn

    apcer = fp / num_spoof if num_spoof else 0
    bpcer = fn / num_live if num_live else 0
    acer = (apcer + bpcer) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
    }


@torch.no_grad()
def evaluate_model(checkpoint_path=BEST_MODEL_PATH):
    """
    Evaluate a trained model checkpoint on the test dataset.
    """

    device = get_device()

    # Load model
    model, checkpoint = load_checkpoint_for_eval(checkpoint_path)
    model = model.to(device)

    # Load data
    _, test_loader = build_dataloaders()

    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)

    print("\n===== Evaluation Results =====\n")

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\nConfusion Matrix")
    print(f"TP: {metrics['tp']}")
    print(f"TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")

    print("\nAnti-Spoofing Metrics")
    print(f"APCER: {metrics['apcer']:.4f}")
    print(f"BPCER: {metrics['bpcer']:.4f}")
    print(f"ACER : {metrics['acer']:.4f}")

    print("\nCheckpoint Info")
    print(f"Epoch trained: {checkpoint.get('epoch')}")
    print(f"Training metrics: {checkpoint.get('metrics')}")

    return metrics