"""HydroRIVERS graph construction extracted without algorithm changes."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass
class RiverGraphResult:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    river_attributes: pd.DataFrame
    snapped_stations: gpd.GeoDataFrame
    edges: list[tuple[int, int, float, bool]]
    isolated_nodes: list[int]
    hydrorivers_numeric_columns: list[str]
    position_to_hyriv: dict[str, int]
    hyriv_to_position: dict[int, str]


def trace_downstream_station(
    start_hyriv_id: int,
    next_down_map: dict[int, int],
    length_map: dict[int, float],
    hyriv_to_position: dict[int, str],
    max_downstream_hops: int,
) -> tuple[str | None, float | None]:
    """Follow NEXT_DOWN until the nearest downstream station is found."""
    current = next_down_map.get(start_hyriv_id, None)
    total_length_km = 0.0
    hops = 0
    visited: set[int] = set()
    while (
        current is not None
        and current != 0
        and hops < max_downstream_hops
    ):
        if current in visited:
            break
        visited.add(current)
        total_length_km += length_map.get(current, 0.0) or 0.0
        if current in hyriv_to_position:
            return hyriv_to_position[current], total_length_km
        current = next_down_map.get(current, None)
        hops += 1
    return None, None


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    value = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    return float(2 * radius * np.arcsin(np.sqrt(value)))


def build_river_graph(
    hydrorivers: gpd.GeoDataFrame,
    coordinate_index: pd.DataFrame,
    node_order: list[str],
    node_to_index: dict[str, int],
    max_downstream_hops: int,
) -> RiverGraphResult:
    """Build the exact river/fallback graph used by the notebook baseline."""
    coordinate_clean = (
        coordinate_index.reset_index()
        .dropna(subset=["latitude", "longitude"])
    )
    stations_gdf = gpd.GeoDataFrame(
        coordinate_clean,
        geometry=gpd.points_from_xy(
            coordinate_clean["longitude"], coordinate_clean["latitude"]
        ),
        crs="EPSG:4326",
    )

    if hydrorivers.crs is None:
        hydrorivers = hydrorivers.set_crs("EPSG:4326")

    hydrorivers_m = hydrorivers.to_crs("EPSG:3857")
    stations_m = stations_gdf.to_crs("EPSG:3857")

    hydro_columns = [
        column for column in hydrorivers.columns if column != "geometry"
    ]
    hydrorivers_m_subset = hydrorivers_m[
        hydro_columns + ["geometry"]
    ].reset_index(drop=True)
    stations_m = stations_m.reset_index(drop=True)

    snapped = gpd.sjoin_nearest(
        stations_m,
        hydrorivers_m_subset,
        how="left",
        distance_col="dist_to_river_m",
    )
    snapped = snapped.drop(columns=["index_right"], errors="ignore")
    snapped = snapped.drop_duplicates(subset="nama_pos", keep="first")

    position_to_hyriv = dict(zip(snapped["nama_pos"], snapped["HYRIV_ID"]))
    hyriv_to_position = {
        hyriv_id: position
        for position, hyriv_id in position_to_hyriv.items()
    }

    next_down_map = dict(
        zip(hydrorivers["HYRIV_ID"], hydrorivers["NEXT_DOWN"])
    )
    length_map = dict(
        zip(hydrorivers["HYRIV_ID"], hydrorivers["LENGTH_KM"])
    )

    edges: list[tuple[int, int, float, bool]] = []
    connected_pairs: set[frozenset[str]] = set()
    for position_a in node_order:
        hyriv_a = position_to_hyriv.get(position_a)
        if hyriv_a is None:
            continue
        position_b, distance_km = trace_downstream_station(
            hyriv_a,
            next_down_map,
            length_map,
            hyriv_to_position,
            max_downstream_hops,
        )
        if position_b is not None and position_b != position_a:
            edges.append(
                (
                    node_to_index[position_a],
                    node_to_index[position_b],
                    distance_km,
                    False,
                )
            )
            connected_pairs.add(frozenset((position_a, position_b)))

    connected_nodes: set[int] = set()
    for source, target, _weight, _fallback in edges:
        connected_nodes.add(source)
        connected_nodes.add(target)

    isolated_nodes = [
        index for index in range(len(node_order)) if index not in connected_nodes
    ]
    for index in isolated_nodes:
        position_i = node_order[index]
        lat_i, lon_i = coordinate_index.loc[
            position_i, ["latitude", "longitude"]
        ]
        distances: list[tuple[float, int]] = []
        for neighbor_index, position_j in enumerate(node_order):
            if neighbor_index == index:
                continue
            lat_j, lon_j = coordinate_index.loc[
                position_j, ["latitude", "longitude"]
            ]
            distances.append(
                (
                    haversine_km(lat_i, lon_i, lat_j, lon_j),
                    neighbor_index,
                )
            )
        distances.sort()
        nearest_index = distances[0][1]
        edges.append((index, nearest_index, distances[0][0], True))

    source_list: list[int] = []
    target_list: list[int] = []
    weight_list: list[float] = []
    for source, target, weight, _fallback in edges:
        source_list += [source, target]
        target_list += [target, source]
        weight_list += [weight, weight]

    edge_index = np.array(
        [source_list, target_list], dtype=np.int64
    )
    edge_weight = np.array(weight_list, dtype=np.float32)

    hydrorivers_numeric_columns = [
        column
        for column in hydrorivers.columns
        if column not in ("geometry", "HYRIV_ID", "NEXT_DOWN")
        and pd.api.types.is_numeric_dtype(hydrorivers[column])
    ]
    river_attributes = (
        snapped.set_index("nama_pos")[hydrorivers_numeric_columns]
        .reindex(node_order)
    )
    river_attributes.columns = [
        f"river_{column.lower()}"
        for column in river_attributes.columns
    ]

    return RiverGraphResult(
        edge_index=edge_index,
        edge_weight=edge_weight,
        river_attributes=river_attributes,
        snapped_stations=snapped,
        edges=edges,
        isolated_nodes=isolated_nodes,
        hydrorivers_numeric_columns=hydrorivers_numeric_columns,
        position_to_hyriv=position_to_hyriv,
        hyriv_to_position=hyriv_to_position,
    )

