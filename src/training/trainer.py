"""Training orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.utils.checkpointing import save_checkpoint
from src.utils.logging import get_logger, log_epoch

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler,
        run_dir: Path,
        device: str,
        num_classes: int,
        early_stopping_patience: int = 8,
        mixed_precision: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.run_dir = Path(run_dir)
        self.device = device
        self.num_classes = num_classes
        self.patience = early_stopping_patience
        self.mixed_precision = mixed_precision and (device == "cuda")

        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None

        self.best_val_f1 = -1.0
        self.epochs_no_improve = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> list[dict]:
        self.model.to(self.device)

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_f1 = val_metrics["macro_f1"]
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)},
            }
            self.history.append(epoch_record)
            log_epoch(self.run_dir, epoch_record)

            logger.info(
                "Epoch %d/%d — loss: %.4f  val_f1: %.4f  val_acc: %.4f",
                epoch, epochs, train_loss, val_f1, val_metrics["accuracy"],
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                save_checkpoint(self.model, self.run_dir / "best.pt")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

            save_checkpoint(self.model, self.run_dir / "last.pt")

        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * len(labels)

        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                logits = self.model(features)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        return compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=self.num_classes,
        )

    # ------------------------------------------------------------------
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                logits = self.model(features)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        return np.array(all_preds), np.array(all_labels)
