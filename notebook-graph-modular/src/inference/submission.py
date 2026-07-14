"""Sample-submission parsing, prediction assembly, and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_sample_submission(path: str | Path) -> pd.DataFrame:
    sample = pd.read_csv(path)
    sample["datetime"] = pd.to_datetime(sample["id"].str[:19])
    sample["nama_pos"] = sample["id"].str[22:]
    return sample


def predictions_to_frame(
    predictions: dict[pd.Timestamp, np.ndarray],
    node_order: list[str],
) -> pd.DataFrame:
    for timestamp, prediction in predictions.items():
        if np.asarray(prediction).shape != (len(node_order),):
            raise ValueError(
                f"Prediksi {timestamp} tidak sejajar dengan node_order: "
                f"{np.asarray(prediction).shape} != ({len(node_order)},)"
            )
    frame = pd.DataFrame(predictions).T
    frame.columns = node_order
    frame.index.name = "datetime"
    return frame.sort_index()


def build_submission(
    sample_submission: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if sample_submission["id"].duplicated().any():
        raise ValueError("sample_submission memiliki id duplikat.")
    prediction_long = prediction_frame.reset_index().melt(
        id_vars="datetime",
        var_name="nama_pos",
        value_name="tma_mdpl_pred",
    )
    submission = sample_submission.merge(
        prediction_long,
        on=["datetime", "nama_pos"],
        how="left",
    )
    missing_count = int(submission["tma_mdpl_pred"].isna().sum())
    if missing_count > 0:
        submission = submission.sort_values(["nama_pos", "datetime"])
        submission["tma_mdpl_pred"] = submission.groupby(
            "nama_pos"
        )["tma_mdpl_pred"].ffill()
        fallback_mean = submission.groupby("nama_pos")[
            "tma_mdpl_pred"
        ].transform("mean")
        submission["tma_mdpl_pred"] = submission[
            "tma_mdpl_pred"
        ].fillna(fallback_mean)

    final = sample_submission[["id"]].merge(
        submission[["id", "tma_mdpl_pred"]],
        on="id",
        how="left",
    )
    final = final.rename(columns={"tma_mdpl_pred": "tma_mdpl"})
    if final.shape[0] != sample_submission.shape[0]:
        raise ValueError("Jumlah baris tidak cocok dengan sample_submission.")
    if final["id"].tolist() != sample_submission["id"].tolist():
        raise ValueError("Urutan id submission berubah dari sample_submission.")
    if final["tma_mdpl"].isna().any():
        raise ValueError("Masih ada NaN di kolom prediksi.")
    if not np.isfinite(final["tma_mdpl"].to_numpy()).all():
        raise ValueError("Prediksi submission memiliki nilai non-finite.")
    return final, missing_count


def save_submission(submission: pd.DataFrame, path: str | Path) -> None:
    submission.to_csv(path, index=False)
