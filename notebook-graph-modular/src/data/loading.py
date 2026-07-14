"""File loading helpers with explicit schema checks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


TRAIN_COLUMNS = ("datetime", "nama_pos", "tma_mdpl")
COORDINATE_COLUMNS = ("nama_pos", "latitude", "longitude")
ENVIRONMENT_KEY_COLUMNS = ("datetime", "nama_pos")
HYDRORIVERS_COLUMNS = ("HYRIV_ID", "NEXT_DOWN", "LENGTH_KM")
GRAPH_DATASET_KEYS = (
    "X_train",
    "y_train",
    "edge_index",
    "edge_weight",
    "node_order",
    "feature_cols",
    "dt_train",
    "panel_arr",
    "obs_datetimes",
)


def _resolve(root: str | Path, relative_path: str | Path) -> Path:
    """Join a dataset root and relative path without requiring a trailing slash."""
    return Path(root).expanduser() / Path(relative_path)


def _validate_required_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} tidak memiliki kolom wajib: {missing}")


def _validate_unique_keys(
    frame: pd.DataFrame,
    keys: list[str],
    dataset_name: str,
) -> None:
    duplicate_mask = frame.duplicated(keys, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, keys].head(5).to_dict("records")
        raise ValueError(
            f"{dataset_name} memiliki key duplikat {keys}; contoh: {examples}"
        )


def load_hydrorivers(
    root: str | Path,
    relative_path: str = "data_pendukung/HydroRIVERS_v10_au_shp/HydroRIVERS_v10_au.shp",
) -> "gpd.GeoDataFrame":
    import geopandas as gpd

    frame = gpd.read_file(_resolve(root, relative_path))
    _validate_required_columns(frame, HYDRORIVERS_COLUMNS, "HydroRIVERS")
    if frame.empty:
        raise ValueError("HydroRIVERS kosong.")
    return frame


def load_station_coordinates(
    root: str | Path,
    relative_path: str = "data_pendukung/koordinat_pos.csv",
) -> pd.DataFrame:
    frame = pd.read_csv(_resolve(root, relative_path))
    _validate_required_columns(frame, COORDINATE_COLUMNS, "koordinat_pos.csv")
    _validate_unique_keys(frame, ["nama_pos"], "koordinat_pos.csv")
    if frame[["latitude", "longitude"]].isna().any().any():
        raise ValueError("koordinat_pos.csv memiliki latitude/longitude kosong.")
    if not frame["latitude"].between(-90, 90).all():
        raise ValueError("koordinat_pos.csv memiliki latitude di luar [-90, 90].")
    if not frame["longitude"].between(-180, 180).all():
        raise ValueError("koordinat_pos.csv memiliki longitude di luar [-180, 180].")
    return frame


def load_train_data(
    root: str | Path,
    relative_path: str = "train.csv",
) -> pd.DataFrame:
    frame = pd.read_csv(_resolve(root, relative_path))
    _validate_required_columns(frame, TRAIN_COLUMNS, "train.csv")
    _validate_unique_keys(frame, ["datetime", "nama_pos"], "train.csv")
    return frame


def load_environment_data(
    root: str | Path,
    feature_columns: Iterable[str] | None = None,
    relative_path: str = "data_pendukung/data_lingkungan.csv",
) -> pd.DataFrame:
    frame = pd.read_csv(_resolve(root, relative_path))
    required = list(ENVIRONMENT_KEY_COLUMNS)
    if feature_columns is not None:
        required.extend(feature_columns)
    _validate_required_columns(frame, required, "data_lingkungan.csv")
    _validate_unique_keys(frame, ["datetime", "nama_pos"], "data_lingkungan.csv")
    return frame


def load_numpy_dataset(
    path: str | Path,
    required_keys: Iterable[str] = GRAPH_DATASET_KEYS,
) -> dict[str, np.ndarray]:
    """Load the notebook NPZ artifact and detach arrays from the open archive."""
    with np.load(Path(path).expanduser(), allow_pickle=True) as archive:
        missing = sorted(set(required_keys) - set(archive.files))
        if missing:
            raise ValueError(f"Dataset NPZ tidak memiliki key wajib: {missing}")
        return {key: archive[key] for key in archive.files}
