"""Sample-submission parsing, prediction assembly, and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_sample_submission(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"sample_submission.csv tidak ditemukan: {source}")
    sample = pd.read_csv(source)
    if "id" not in sample.columns:
        raise ValueError(f"sample_submission.csv tidak memiliki kolom wajib 'id': {source}")
    if sample.empty:
        raise ValueError(f"sample_submission.csv kosong: {source}")
    if sample["id"].isna().any() or sample["id"].duplicated().any():
        raise ValueError("sample_submission.csv memiliki id kosong atau duplikat.")
    identifier = sample["id"].astype(str)
    parsed = identifier.str.extract(
        r"^(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (?P<nama_pos>.+)$"
    )
    if parsed.isna().any().any():
        examples = identifier[parsed.isna().any(axis=1)].head(3).tolist()
        raise ValueError(
            "Format id sample_submission tidak valid; mengharapkan "
            f"'YYYY-MM-DD HH:MM:SS - nama_pos'. Contoh: {examples}"
        )
    sample["datetime"] = pd.to_datetime(parsed["datetime"], errors="raise")
    sample["nama_pos"] = parsed["nama_pos"].to_numpy()
    return sample


def predictions_to_frame(
    predictions: dict[pd.Timestamp, np.ndarray],
    node_order: list[str],
) -> pd.DataFrame:
    if not node_order or len(set(node_order)) != len(node_order):
        raise ValueError("node_order prediction harus tidak kosong dan unik.")
    if not predictions:
        raise ValueError("Tidak ada prediction yang tersedia untuk membuat submission.")
    for timestamp, prediction in predictions.items():
        values = np.asarray(prediction)
        if values.shape != (len(node_order),):
            raise ValueError(
                f"Prediksi {timestamp} tidak sejajar dengan node_order: "
                f"{values.shape} != ({len(node_order)},)"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Prediksi {timestamp} mengandung NaN atau infinity.")
    frame = pd.DataFrame(predictions).T
    frame.columns = node_order
    frame.index.name = "datetime"
    return frame.sort_index()


def build_submission(
    sample_submission: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required_sample_columns = {"id", "datetime", "nama_pos"}
    missing_sample_columns = sorted(required_sample_columns - set(sample_submission))
    if missing_sample_columns:
        raise ValueError(
            "sample_submission yang diparse tidak lengkap; kolom hilang: "
            f"{missing_sample_columns}"
        )
    if prediction_frame.index.name != "datetime":
        raise ValueError("prediction_frame harus memiliki index bernama 'datetime'.")
    if prediction_frame.columns.duplicated().any():
        raise ValueError("prediction_frame memiliki node/kolom duplikat.")
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
        unavailable_stations = sorted(
            submission.loc[fallback_mean.isna(), "nama_pos"].unique().tolist()
        )
        if unavailable_stations:
            raise ValueError(
                "Prediction tidak mencakup station sample_submission berikut: "
                f"{unavailable_stations}"
            )
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
        raise ValueError(
            "Jumlah baris submission tidak cocok dengan sample_submission: "
            f"{final.shape[0]} != {sample_submission.shape[0]}"
        )
    if final["id"].tolist() != sample_submission["id"].tolist():
        raise ValueError("Urutan id submission berubah dari sample_submission.")
    if final["tma_mdpl"].isna().any():
        raise ValueError("Masih ada NaN di kolom prediksi.")
    if not np.isfinite(final["tma_mdpl"].to_numpy()).all():
        raise ValueError("Prediksi submission memiliki nilai non-finite.")
    return final, missing_count


def save_submission(submission: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
