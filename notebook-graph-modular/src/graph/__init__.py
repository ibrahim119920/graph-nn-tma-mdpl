"""Graph construction, adjacency, and graph-aware dataset utilities."""

from .adjacency import build_normalized_adjacency

__all__ = [
    "RiverGraphResult",
    "build_normalized_adjacency",
    "build_river_graph",
    "haversine_km",
    "trace_downstream_station",
]


def __getattr__(name: str):
    """Load geospatial/PyTorch helpers only when requested."""
    if name in {
        "RiverGraphResult",
        "build_river_graph",
        "haversine_km",
        "trace_downstream_station",
    }:
        from .construction import (
            RiverGraphResult,
            build_river_graph,
            haversine_km,
            trace_downstream_station,
        )

        return {
            "RiverGraphResult": RiverGraphResult,
            "build_river_graph": build_river_graph,
            "haversine_km": haversine_km,
            "trace_downstream_station": trace_downstream_station,
        }[name]
    if name in {
        "GraphTimeSeriesDataset",
        "build_graph_dataloaders",
    }:
        from .dataset import GraphTimeSeriesDataset, build_graph_dataloaders

        return {
            "GraphTimeSeriesDataset": GraphTimeSeriesDataset,
            "build_graph_dataloaders": build_graph_dataloaders,
        }[name]
    raise AttributeError(name)
