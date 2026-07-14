import tempfile
import unittest
from pathlib import Path

from src.utils.artifacts import ExperimentArtifactStore
from src.utils.config import load_project_config
from src.utils.logging import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigAndArtifactTest(unittest.TestCase):
    def test_default_config_preserves_baseline(self):
        config = load_project_config(PROJECT_ROOT / "configs/default.json")
        self.assertEqual(config.model.gcn_hidden, 64)
        self.assertEqual(config.model.gru_hidden, 64)
        self.assertTrue(config.model.spatial_residual)
        self.assertEqual(config.training.epochs, 100)
        self.assertEqual(config.inference.n_lags, 11)
        self.assertTrue(config.paths.dataset.is_absolute())

    def test_artifact_store_writes_manifest_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentArtifactStore(
                directory, "test-project", run_id="test-run"
            )
            store.write_json("metrics.json", {"rmse": 0.5})
            store.update_manifest(status="completed")
            self.assertTrue((store.run_dir / "metrics.json").exists())
            manifest = (store.run_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
            self.assertIn('"status": "completed"', manifest)

    def test_logging_writes_run_file(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = configure_logging(
                "test-logger", directory, "test-run", "INFO"
            )
            logger.info("infrastructure-ready")
            for handler in logger.handlers:
                handler.flush()
            content = (Path(directory) / "test-run.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("infrastructure-ready", content)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
