"""Reusable epoch and fit loops preserving the notebook update order."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping


class ImprovementCallback(Protocol):
    def on_improvement(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
    ) -> None: ...


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    teacher_validation_loss: float
    rollout_validation_loss: float
    learning_rate: float
    duration_seconds: float
    improved: bool


@dataclass
class TrainingResult:
    records: list[EpochRecord] = field(default_factory=list)
    best_metric: float = float("inf")
    stopped_early: bool = False

    def history_dict(self) -> dict[str, list[float]]:
        return {
            "train": [record.train_loss for record in self.records],
            "teacher_val": [
                record.teacher_validation_loss for record in self.records
            ],
            "rollout_val": [
                record.rollout_validation_loss for record in self.records
            ],
            "lr": [record.learning_rate for record in self.records],
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    zero_grad_set_to_none: bool | None = None,
) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.set_grad_enabled(training):
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            if training:
                if zero_grad_set_to_none is None:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(
                        set_to_none=zero_grad_set_to_none
                    )
            prediction = model(feature_batch)
            loss = criterion(prediction, target_batch)

            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()

            total_loss += loss.item()
            batch_count += 1
    return total_loss / max(batch_count, 1)


def fit_with_rollout(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device | str,
    *,
    epochs: int,
    early_stopping: EarlyStopping,
    rollout_metric: Callable[[], float],
    improvement_callback: ImprovementCallback | None = None,
    gradient_clip_norm: float | None = None,
    zero_grad_set_to_none: bool | None = None,
    epoch_callback: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    result = TrainingResult()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            zero_grad_set_to_none=zero_grad_set_to_none,
        )
        teacher_validation_loss = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        rollout_validation_loss = rollout_metric()
        scheduler.step(rollout_validation_loss)
        duration = time.time() - start_time

        improved, should_stop = early_stopping.update(
            rollout_validation_loss
        )
        if improved and improvement_callback is not None:
            improvement_callback.on_improvement(
                model, epoch, rollout_validation_loss
            )

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            teacher_validation_loss=teacher_validation_loss,
            rollout_validation_loss=rollout_validation_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
            duration_seconds=duration,
            improved=improved,
        )
        result.records.append(record)
        result.best_metric = early_stopping.best_metric
        if epoch_callback is not None:
            epoch_callback(record)

        if should_stop:
            result.stopped_early = True
            break
    return result
