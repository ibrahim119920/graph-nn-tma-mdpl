"""File loading helpers with explicit schema checks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


TRAIN_COLUMNS = ("datetime", "nama_pos", "tma_mdpl")
TEST_COLUMNS = ("id",)
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


def load_test_data(
    root: str | Path,
    relative_path: str = "test.csv",
) -> pd.DataFrame:
    frame = pd.read_csv(_resolve(root, relative_path))
    _validate_required_columns(frame, TEST_COLUMNS, "test.csv")
    if frame.empty:
        raise ValueError("test.csv kosong.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("test.csv memiliki id kosong atau duplikat.")
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
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset NPZ tidak ditemukan: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as archive:
        missing = sorted(set(required_keys) - set(archive.files))
        if missing:
            raise ValueError(
                f"Dataset NPZ {dataset_path} tidak memiliki key wajib: {missing}"
            )
        dataset = {key: archive[key] for key in archive.files}
    validate_graph_dataset(dataset, dataset_path)
    return dataset


def validate_graph_dataset(
    dataset: dict[str, np.ndarray],
    dataset_path: str | Path = "<in-memory>",
) -> None:
    """Validate the serialized tensor contract before train/evaluate/inference."""
    label = str(dataset_path)
    x_train = np.asarray(dataset["X_train"])
    y_train = np.asarray(dataset["y_train"])
    edge_index = np.asarray(dataset["edge_index"])
    edge_weight = np.asarray(dataset["edge_weight"])
    node_order = np.asarray(dataset["node_order"])
    feature_columns = np.asarray(dataset["feature_cols"])
    panel = np.asarray(dataset["panel_arr"])
    datetimes = np.asarray(dataset["dt_train"])
    observation_datetimes = np.asarray(dataset["obs_datetimes"])

    if x_train.ndim != 4:
        raise ValueError(f"X_train pada {label} harus 4D (N,T,nodes,features), got {x_train.shape}.")
    if y_train.ndim != 2:
        raise ValueError(f"y_train pada {label} harus 2D (N,nodes), got {y_train.shape}.")
    if x_train.shape[0] == 0:
        raise ValueError(f"X_train pada {label} kosong; tidak ada window training.")
    if x_train.shape[:1] != y_train.shape[:1] or x_train.shape[2] != y_train.shape[1]:
        raise ValueError(
            f"Shape X_train {x_train.shape} dan y_train {y_train.shape} tidak sejajar pada {label}."
        )
    if node_order.ndim != 1 or len(node_order) != x_train.shape[2]:
        raise ValueError(
            f"node_order pada {label} harus sepanjang {x_train.shape[2]}, got {node_order.shape}."
        )
    if feature_columns.ndim != 1 or len(feature_columns) != x_train.shape[3]:
        raise ValueError(
            f"feature_cols pada {label} harus sepanjang {x_train.shape[3]}, got {feature_columns.shape}."
        )
    if len(set(node_order.tolist())) != len(node_order):
        raise ValueError(f"node_order pada {label} memiliki identifier duplikat.")
    if len(set(feature_columns.tolist())) != len(feature_columns):
        raise ValueError(f"feature_cols pada {label} memiliki nama fitur duplikat.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index pada {label} harus berbentuk (2,E), got {edge_index.shape}.")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weight pada {label} harus sepanjang {edge_index.shape[1]}, got {edge_weight.shape}."
        )
    if edge_index.size and (
        edge_index.min() < 0 or edge_index.max() >= len(node_order)
    ):
        raise ValueError(f"edge_index pada {label} memiliki node di luar node_order.")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError(f"X_train atau y_train pada {label} memiliki NaN/infinity.")
    if not np.isfinite(edge_weight).all() or (edge_weight < 0).any():
        raise ValueError(f"edge_weight pada {label} harus finite dan non-negatif.")
    if panel.ndim != 3 or panel.shape[1:] != x_train.shape[2:]:
        raise ValueError(
            f"panel_arr pada {label} harus (time,{x_train.shape[2]},{x_train.shape[3]}), got {panel.shape}."
        )
    if len(datetimes) != x_train.shape[0]:
        raise ValueError(f"dt_train pada {label} tidak sejajar dengan jumlah sample X_train.")
    if len(observation_datetimes) != panel.shape[0]:
        raise ValueError(f"obs_datetimes pada {label} tidak sejajar dengan panel_arr.")
