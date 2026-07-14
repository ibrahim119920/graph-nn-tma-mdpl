"""Configured project launcher with logging and experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_checkpoint
from scripts.make_submission import make_submission
from scripts.train import train_baseline
from src.utils import (
    ExperimentArtifactStore,
    configure_logging,
    load_project_config,
)


VALID_STAGES = {"build", "train", "evaluate", "submission", "pipeline"}


def run_from_config(
    config_path: str | Path,
    stage: str = "pipeline",
) -> dict:
    if stage not in VALID_STAGES:
        raise ValueError(f"Stage tidak valid: {stage}")
    config = load_project_config(config_path)
    for path in (
        config.paths.checkpoint.parent,
        config.paths.submission.parent,
        config.paths.artifact_dir,
        config.paths.log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    artifacts = ExperimentArtifactStore(
        config.paths.artifact_dir, config.project_name
    )
    logger = configure_logging(
        config.project_name,
        config.paths.log_dir,
        artifacts.run_id,
        config.logging.level,
    )
    artifacts.write_json("config.json", config.raw)
    artifacts.update_manifest(
        stage=stage,
        config_path=str(config.source_path),
        status="running",
    )
    logger.info("Run %s dimulai dengan stage=%s", artifacts.run_id, stage)
    result: dict = {
        "run_id": artifacts.run_id,
        "artifact_dir": artifacts.run_dir,
    }
    try:
        if stage in {"build", "pipeline"}:
            from scripts.build_dataset import build_dataset_artifact

            dataset_summary = build_dataset_artifact(
                config.paths.raw_data_root,
                config.paths.dataset,
                config,
                logger=logger,
            )
            result["dataset"] = dataset_summary
            artifacts.write_json("dataset_summary.json", dataset_summary)

        if stage in {"train", "pipeline"}:
            training = train_baseline(
                config.paths.dataset,
                config.paths.checkpoint,
                config=config,
                logger=logger,
            )
            result["training"] = training
            artifacts.write_json(
                "training_history.json",
                training["result"].history_dict(),
            )

        if stage in {"evaluate", "pipeline"}:
            evaluation = evaluate_checkpoint(
                config.paths.dataset,
                config.paths.checkpoint,
                config=config,
                logger=logger,
            )
            result["evaluation"] = evaluation
            artifacts.write_json(
                "evaluation_metrics.json",
                {
                    "rmse": evaluation.rmse,
                    "mae": evaluation.mae,
                    "per_node_rmse": evaluation.per_node_rmse,
                },
            )

        if stage in {"submission", "pipeline"}:
            submission, inference = make_submission(
                config.paths.dataset,
                config.paths.checkpoint,
                config.paths.sample_submission,
                config.paths.submission,
                config=config,
                logger=logger,
            )
            result["submission"] = submission
            result["inference"] = inference
            artifacts.write_json(
                "submission_summary.json",
                {
                    "path": config.paths.submission,
                    "rows": submission.shape[0],
                    "predicted_timesteps": len(inference.predictions),
                    "skipped_timesteps": len(inference.skipped_datetimes),
                    "filled_training_cells": inference.filled_training_cells,
                },
            )

        artifacts.update_manifest(
            status="completed",
            checkpoint=str(config.paths.checkpoint),
            submission=str(config.paths.submission),
        )
        logger.info("Run %s selesai", artifacts.run_id)
        return result
    except Exception as error:
        artifacts.update_manifest(
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        logger.exception("Run %s gagal", artifacts.run_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.json"),
    )
    parser.add_argument(
        "--stage",
        choices=sorted(VALID_STAGES),
        default="pipeline",
    )
    args = parser.parse_args()
    run_from_config(args.config, args.stage)


if __name__ == "__main__":
    main()
