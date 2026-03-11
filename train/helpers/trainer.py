from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from train.helpers.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_training_components(model: torch.nn.Module):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return criterion, optimizer, scheduler


def compute_classification_metrics(all_labels: List[int], all_preds: List[int]) -> Dict[str, float]:
    """
    Label convention:
    0 = spoof
    1 = live
    """

    tp = sum((p == 1 and y == 1) for y, p in zip(all_labels, all_preds))  # live predicted live
    tn = sum((p == 0 and y == 0) for y, p in zip(all_labels, all_preds))  # spoof predicted spoof
    fp = sum((p == 1 and y == 0) for y, p in zip(all_labels, all_preds))  # spoof predicted live
    fn = sum((p == 0 and y == 1) for y, p in zip(all_labels, all_preds))  # live predicted spoof

    total = max(len(all_labels), 1)

    accuracy = (tp + tn) / total

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Anti-spoofing metrics
    # APCER = Attack Presentation Classification Error Rate
    #       = spoof samples classified as live / all spoof samples
    # BPCER = Bona Fide Presentation Classification Error Rate
    #       = live samples classified as spoof / all live samples
    # ACER = (APCER + BPCER) / 2

    num_spoof = tn + fp   # actual spoof
    num_live = tp + fn    # actual live

    apcer = fp / num_spoof if num_spoof > 0 else 0.0
    bpcer = fn / num_live if num_live > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total_samples
    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = epoch_loss

    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        },
        path,
    )


def fit(model, train_loader, test_loader, num_epochs=NUM_EPOCHS):
    device = get_device()
    model = model.to(device)

    criterion, optimizer, scheduler = build_training_components(model)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_test_acc = 0.0

    print(f"Using device: {device}")
    print(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"APCER: {test_metrics['apcer']:.4f} | "
            f"BPCER: {test_metrics['bpcer']:.4f} | "
            f"ACER: {test_metrics['acer']:.4f}"
        )

        print(
            f"Confusion Matrix -> "
            f"TP: {int(test_metrics['tp'])}, "
            f"TN: {int(test_metrics['tn'])}, "
            f"FP: {int(test_metrics['fp'])}, "
            f"FN: {int(test_metrics['fn'])}"
        )

        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics={
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_tp": test_metrics["tp"],
                "test_tn": test_metrics["tn"],
                "test_fp": test_metrics["fp"],
                "test_fn": test_metrics["fn"],
                "test_apcer": test_metrics["apcer"],
                "test_bpcer": test_metrics["bpcer"],
                "test_acer": test_metrics["acer"],
            },
        )

        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = test_metrics["accuracy"]

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "test_tp": test_metrics["tp"],
                    "test_tn": test_metrics["tn"],
                    "test_fp": test_metrics["fp"],
                    "test_fn": test_metrics["fn"],
                    "test_apcer": test_metrics["apcer"],
                    "test_bpcer": test_metrics["bpcer"],
                    "test_acer": test_metrics["acer"],
                },
            )

            print(f"Best model updated at epoch {epoch + 1} with test acc {best_test_acc:.4f}")

    print("Training completed.")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Last model saved to: {LAST_MODEL_PATH}")