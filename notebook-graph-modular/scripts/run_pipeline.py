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
from src.inference import summarize_submission
from src.utils import (
    ExperimentArtifactStore,
    configure_logging,
    load_project_config,
    seed_everything,
    validate_stage_paths,
)


VALID_STAGES = {"build", "train", "evaluate", "submission", "pipeline"}


def run_from_config(
    config_path: str | Path,
    stage: str = "pipeline",
    dry_run: bool = False,
) -> dict:
    if stage not in VALID_STAGES:
        raise ValueError(f"Stage tidak valid: {stage}")
    config = load_project_config(config_path)
    validate_stage_paths(config, stage)
    _validate_stage_contract(config, stage)
    if dry_run:
        from scripts.smoke_test import run_smoke_test

        smoke_shape = run_smoke_test()
        return {
            "dry_run": True,
            "stage": stage,
            "config_path": config.source_path,
            "workspace_root": config.workspace_root,
            "smoke_output_shape": smoke_shape,
        }
    seed_everything(config.runtime.seed)
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
    logger.info("Seed: %s", config.runtime.seed)
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
                training["result"].history_dict(
                    epochs_requested=config.training.epochs
                ),
            )
            artifacts.write_json(
                "checkpoint_summary.json", training["checkpoint_metadata"]
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
            submission_summary = summarize_submission(submission)
            submission_summary.update(
                {
                    "path": config.paths.submission,
                    "predicted_timesteps": len(inference.predictions),
                    "skipped_timesteps": len(inference.skipped_datetimes),
                    "filled_training_cells": inference.filled_training_cells,
                    "fallback_rows": inference.fallback_count,
                }
            )
            artifacts.write_json(
                "submission_summary.json", submission_summary
            )

        manifest_updates = {
            "status": "completed",
            "checkpoint": str(config.paths.checkpoint),
            "submission": str(config.paths.submission),
        }
        if "training" in result:
            training_result = result["training"]["result"]
            manifest_updates.update(
                {
                    "epochs_ran": len(training_result.records),
                    "best_epoch": training_result.best_epoch,
                    "best_rollout_validation_rmse_normalized": (
                        training_result.best_metric
                    ),
                    "stopped_early": training_result.stopped_early,
                    "stop_reason": (
                        "early_stopping"
                        if training_result.stopped_early
                        else "max_epochs_reached"
                    ),
                }
            )
        artifacts.update_manifest(**manifest_updates)
        logger.info("Run %s selesai", artifacts.run_id)
        return result
    except Exception as error:
        artifacts.update_manifest(
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        logger.exception("Run %s gagal", artifacts.run_id)
        raise


def _validate_stage_contract(config, stage: str) -> None:
    """Load lightweight schemas before a long-running stage starts."""
    if stage in {"build", "pipeline"}:
        from src.data.loading import (
            load_environment_data,
            load_station_coordinates,
            load_test_data,
            load_train_data,
        )
        from src.features.environment import DEFAULT_ENVIRONMENT_FEATURES

        # These reads validate required columns and unique identifiers before
        # graph construction or training starts. HydroRIVERS itself is checked
        # by the preflight path validator and is loaded by the build stage.
        load_train_data(config.paths.raw_data_root)
        load_test_data(config.paths.raw_data_root)
        load_station_coordinates(config.paths.raw_data_root)
        load_environment_data(
            config.paths.raw_data_root, DEFAULT_ENVIRONMENT_FEATURES
        )
    if stage in {"submission", "pipeline"}:
        from src.inference.submission import parse_sample_submission

        parse_sample_submission(config.paths.sample_submission)
    if stage in {"train", "evaluate", "submission"}:
        from src.data.loading import load_numpy_dataset

        dataset = load_numpy_dataset(config.paths.dataset)
        feature_columns = list(dataset["feature_cols"])
        missing_lag_columns = [
            f"wl_t-{lag}"
            for lag in range(1, config.inference.n_lags + 1)
            if f"wl_t-{lag}" not in feature_columns
        ]
        if missing_lag_columns:
            raise ValueError(
                "Dataset feature schema tidak cocok dengan inference.n_lags="
                f"{config.inference.n_lags}; fitur hilang: {missing_lag_columns}"
            )
        dataset_time_window = int(dataset["X_train"].shape[1])
        if dataset_time_window != config.inference.time_window:
            raise ValueError(
                "time_window config dan dataset berbeda: "
                f"{config.inference.time_window} != {dataset_time_window}"
            )
        if stage in {"evaluate", "submission"}:
            from src.training import (
                load_checkpoint,
                validate_checkpoint_dataset_alignment,
                validate_checkpoint_model_compatibility,
            )

            checkpoint = load_checkpoint(config.paths.checkpoint, "cpu")
            validate_checkpoint_dataset_alignment(checkpoint, dataset)
            validate_checkpoint_model_compatibility(
                checkpoint,
                gcn_hidden=config.model.gcn_hidden,
                gru_hidden=config.model.gru_hidden,
                spatial_residual=config.model.spatial_residual,
            )


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validasi config, input path, schema, dan checkpoint tanpa "
            "membangun dataset, melatih model, atau menulis artifact."
        ),
    )
    args = parser.parse_args()
    result = run_from_config(args.config, args.stage, args.dry_run)
    if args.dry_run:
        print(
            "DRY_RUN_OK "
            f"stage={result['stage']} config={result['config_path']} "
            f"smoke_output_shape={result['smoke_output_shape']}"
        )


if __name__ == "__main__":
    main()
