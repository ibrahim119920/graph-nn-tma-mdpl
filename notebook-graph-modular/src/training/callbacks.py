"""Stateful callbacks used by the unchanged rollout-selection methodology."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch.nn as nn

from src.models.utils import unwrap_model

from .checkpoints import save_checkpoint


@dataclass
class EarlyStopping:
    patience: int
    best_metric: float = float("inf")
    counter: int = 0

    def update(self, metric: float) -> tuple[bool, bool]:
        improved = metric < self.best_metric
        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
        return improved, self.counter >= self.patience


@dataclass
class BestCheckpointCallback:
    path: str | Path
    payload_factory: Callable[[nn.Module], dict]

    def on_improvement(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
    ) -> None:
        del epoch, metric
        save_checkpoint(self.path, self.payload_factory(model))


@dataclass
class BestStateCallback:
    state_dict: dict | None = field(default=None, init=False)

    def on_improvement(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
    ) -> None:
        del epoch, metric
        self.state_dict = copy.deepcopy(unwrap_model(model).state_dict())

