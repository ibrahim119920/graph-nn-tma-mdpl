"""Run baseline autoregressive inference and write submission.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.data.loading import load_numpy_dataset
from src.inference import (
    build_submission,
    parse_sample_submission,
    predict_autoregressive,
    predictions_to_frame,
    save_submission,
)
from src.training import (
    load_checkpoint,
    load_temporal_gnn_from_checkpoint,
    validate_checkpoint_dataset_alignment,
)
from src.utils.config import ProjectConfig


def make_submission(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    sample_submission_path: str | Path,
    output_path: str | Path,
    device: torch.device | None = None,
    config: ProjectConfig | None = None,
    logger=None,
):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    data = load_numpy_dataset(dataset_path)
    validate_checkpoint_dataset_alignment(checkpoint, data)
    if config and config.inference.time_window != int(checkpoint["time_window"]):
        raise ValueError(
            "time_window config dan checkpoint berbeda: "
            f"{config.inference.time_window} != {checkpoint['time_window']}"
        )
    model = load_temporal_gnn_from_checkpoint(
        checkpoint,
        device,
        gcn_hidden=config.model.gcn_hidden if config else 64,
        gru_hidden=config.model.gru_hidden if config else 64,
        dropout=0.0,
    )
    sample = parse_sample_submission(sample_submission_path)
    target_datetimes = pd.DatetimeIndex(
        sorted(sample["datetime"].unique())
    )
    result = predict_autoregressive(
        model,
        data["panel_arr"].copy(),
        pd.to_datetime(data["obs_datetimes"]),
        target_datetimes,
        list(data["feature_cols"]),
        train_end=(
            config.inference.train_end
            if config
            else "2025-09-18 18:00:00"
        ),
        n_lags=config.inference.n_lags if config else 11,
        time_window=(
            config.inference.time_window
            if config
            else int(checkpoint["time_window"])
        ),
        feature_mean=checkpoint["feat_mean"],
        feature_std=checkpoint["feat_std"],
        station_center=checkpoint["station_center"],
        station_scale=checkpoint["station_scale"],
        station_low=checkpoint["station_low"],
        station_high=checkpoint["station_high"],
        water_value_positions=list(checkpoint["wl_value_positions"]),
        water_std_positions=list(checkpoint["wl_std_positions"]),
        device=device,
    )
    prediction_frame = predictions_to_frame(
        result.predictions, list(data["node_order"])
    )
    submission, fallback_count = build_submission(sample, prediction_frame)
    save_submission(submission, output_path)
    emit = logger.info if logger else print
    emit(
        f"Berhasil diprediksi: {len(result.predictions)} / "
        f"{len(target_datetimes)} timestep"
    )
    if result.skipped_datetimes:
        emit(f"Dilewati: {len(result.skipped_datetimes)}")
    if fallback_count:
        emit(f"Fallback submission: {fallback_count} baris")
    emit(f"Submission disimpan ke: {output_path}")
    return submission, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.json"),
    )
    args = parser.parse_args()
    from scripts.run_pipeline import run_from_config

    run_from_config(args.config, stage="submission")


if __name__ == "__main__":
    main()
