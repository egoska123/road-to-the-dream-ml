import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def loggits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    return predictions


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        predictions = loggits_to_predictions(logits)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_precision = precision_score(all_targets, all_predictions, zero_division=0)
    epoch_recall = recall_score(all_targets, all_predictions, zero_division=0)
    epoch_f1 = f1_score(all_targets, all_predictions, zero_division=0)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "precision": epoch_precision,
        "recall": epoch_recall,
        "f1": epoch_f1,
    }


def validate_one_epoch(
    model: nn.Module, criterion: torch.nn.Module, dataloader: DataLoader, device: str
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)

            predictions = loggits_to_predictions(logits)

            all_targets.extend(y_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_accuracy = accuracy_score(all_targets, all_predictions)
        epoch_precision = precision_score(all_targets, all_predictions, zero_division=0)
        epoch_recall = recall_score(all_targets, all_predictions, zero_division=0)
        epoch_f1 = f1_score(all_targets, all_predictions, zero_division=0)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "f1": epoch_f1,
        }


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: str,
    num_epochs: int,
    patience: int,
    model_save_path: str,
) -> dict[str, list[float]]:
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_precision": [],
        "valid_recall": [],
        "valid_f1": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model, optimizer, criterion, train_loader, device
        )
        valid_metrics = validate_one_epoch(model, criterion, valid_loader, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])

        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_accuracy"].append(valid_metrics["accuracy"])
        history["valid_precision"].append(valid_metrics["precision"])
        history["valid_recall"].append(valid_metrics["recall"])
        history["valid_f1"].append(valid_metrics["f1"])

        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        print(
            f"Train | loss: {train_metrics['loss']:.4f} | "
            f"acc: {train_metrics['accuracy']:.4f} | "
            f"precision: {train_metrics['precision']:.4f} | "
            f"recall: {train_metrics['recall']:.4f} | "
            f"f1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Valid | loss: {valid_metrics['loss']:.4f} | "
            f"acc: {valid_metrics['accuracy']:.4f} | "
            f"precision: {valid_metrics['precision']:.4f} | "
            f"recall: {valid_metrics['recall']:.4f} | "
            f"f1: {valid_metrics['f1']:.4f}"
        )

        if valid_metrics["loss"] < best_val_loss:
            best_val_loss = valid_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_save_path)
            print("Лучшая модель сохранена")
        else:
            epochs_without_improvement += 1
            print(f"Нет улучшения: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print(f"Нет улучшения в {patience} эпох, обучение остановлено")
                break

    return history
