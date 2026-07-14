"""Structured experiment artifact storage under outputs/experiments."""

from __future__ import annotations

import json
import platform
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Tidak dapat menserialisasi {type(value).__name__}")


class ExperimentArtifactStore:
    def __init__(
        self,
        root: str | Path,
        project_name: str,
        run_id: str | None = None,
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{timestamp}-{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.write_json(
            "manifest.json",
            {
                "run_id": self.run_id,
                "project_name": project_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "status": "running",
            },
        )

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def update_manifest(self, **updates: Any) -> Path:
        path = self.run_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(updates)
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return self.write_json("manifest.json", manifest)

