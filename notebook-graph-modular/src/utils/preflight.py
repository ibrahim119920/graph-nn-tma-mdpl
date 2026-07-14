"""Early path, input, and writable-output validation for CLI stages."""

from __future__ import annotations

import os
from pathlib import Path

from .config import ProjectConfig


RAW_DATA_FILES = {
    "train data": Path("train.csv"),
    "test data": Path("test.csv"),
    "station coordinates": Path("data_pendukung/koordinat_pos.csv"),
    "environment data": Path("data_pendukung/data_lingkungan.csv"),
    "HydroRIVERS shapefile": Path(
        "data_pendukung/HydroRIVERS_v10_au_shp/HydroRIVERS_v10_au.shp"
    ),
}


def require_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} tidak ditemukan: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} bukan file reguler: {candidate}")
    return candidate


def require_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} tidak ditemukan: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{label} bukan direktori: {candidate}")
    return candidate


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_output_paths(config: ProjectConfig) -> None:
    """Reject accidental writes into an immutable input dataset directory."""
    raw_root = config.paths.raw_data_root
    outputs = {
        "dataset": config.paths.dataset,
        "checkpoint": config.paths.checkpoint,
        "submission": config.paths.submission,
        "artifact_dir": config.paths.artifact_dir,
        "log_dir": config.paths.log_dir,
    }
    for label, output in outputs.items():
        if _is_within(output, raw_root):
            raise ValueError(
                f"Path output '{label}' tidak boleh berada di raw_data_root "
                f"(read-only): {output}"
            )
        nearest_parent = output if output.suffix == "" else output.parent
        while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
            nearest_parent = nearest_parent.parent
        if nearest_parent.exists() and not os.access(nearest_parent, os.W_OK):
            raise PermissionError(
                f"Direktori output tidak writable untuk '{label}': {nearest_parent}"
            )


def validate_raw_dataset_layout(config: ProjectConfig) -> None:
    root = require_directory(config.paths.raw_data_root, "Dataset root")
    for label, relative_path in RAW_DATA_FILES.items():
        require_file(root / relative_path, label)


def validate_sample_submission_path(config: ProjectConfig) -> None:
    require_file(config.paths.sample_submission, "sample_submission.csv")


def validate_stage_paths(config: ProjectConfig, stage: str) -> None:
    """Validate only the inputs that must pre-exist for a requested stage."""
    validate_output_paths(config)
    if stage in {"build", "pipeline"}:
        validate_raw_dataset_layout(config)
    if stage in {"submission", "pipeline"}:
        validate_sample_submission_path(config)
    if stage in {"train", "evaluate", "submission"}:
        require_file(config.paths.dataset, "Dataset NPZ")
    if stage in {"evaluate", "submission"}:
        require_file(config.paths.checkpoint, "Checkpoint model")
