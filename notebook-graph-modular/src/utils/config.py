"""Typed JSON configuration loading and path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    raw_data_root: Path
    dataset: Path
    checkpoint: Path
    sample_submission: Path
    submission: Path
    artifact_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    gcn_hidden: int = 64
    gru_hidden: int = 64
    dropout: float = 0.2
    spatial_residual: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    validation_ratio: float = 0.1
    huber_beta: float = 0.5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5


@dataclass(frozen=True)
class InferenceConfig:
    n_lags: int = 11
    time_window: int = 12
    train_end: str = "2025-09-18 18:00:00"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution settings that do not alter the model methodology."""

    seed: int = 2026


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    workspace_root: Path
    paths: PathConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    logging: LoggingConfig
    runtime: RuntimeConfig
    source_path: Path
    raw: dict[str, Any]


def _resolve_path(value: str, workspace_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else workspace_root / path


def load_project_config(path: str | Path) -> ProjectConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan atau bukan file: {source_path}"
        )
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON konfigurasi tidak valid: {source_path} "
            f"(baris {error.lineno}, kolom {error.colno})"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"Root konfigurasi harus object JSON: {source_path}")
    project_root = source_path.parent.parent
    workspace_value = Path(raw.get("workspace_root", ".")).expanduser()
    workspace_root = (
        workspace_value
        if workspace_value.is_absolute()
        else (project_root / workspace_value)
    ).resolve()

    path_values = raw.get("paths")
    if not isinstance(path_values, dict):
        raise ValueError(
            f"Konfigurasi harus memiliki object 'paths': {source_path}"
        )
    required_path_keys = set(PathConfig.__dataclass_fields__)
    missing_path_keys = sorted(required_path_keys - set(path_values))
    if missing_path_keys:
        raise ValueError(
            "Konfigurasi paths tidak lengkap pada "
            f"{source_path}: {missing_path_keys}"
        )
    paths = PathConfig(
        **{
            key: _resolve_path(value, workspace_root)
            for key, value in path_values.items()
        }
    )
    config = ProjectConfig(
        project_name=raw.get("project_name", "hydro-temporal-gnn"),
        workspace_root=workspace_root,
        paths=paths,
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        source_path=source_path,
        raw=raw,
    )
    validate_project_config(config)
    return config


def validate_project_config(config: ProjectConfig) -> None:
    if config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ValueError("batch_size dan epochs harus positif.")
    if not 0 < config.training.validation_ratio < 1:
        raise ValueError("validation_ratio harus berada di antara 0 dan 1.")
    if config.model.gcn_hidden <= 0 or config.model.gru_hidden <= 0:
        raise ValueError("Dimensi hidden model harus positif.")
    if not 0 <= config.model.dropout < 1:
        raise ValueError("dropout harus berada pada [0, 1).")
    if not isinstance(config.model.spatial_residual, bool):
        raise ValueError("model.spatial_residual harus boolean.")
    if config.inference.n_lags <= 0 or config.inference.time_window <= 0:
        raise ValueError("n_lags dan time_window harus positif.")
    if not isinstance(config.runtime.seed, int) or config.runtime.seed < 0:
        raise ValueError("runtime.seed harus bilangan bulat non-negatif.")
