# Hydro Temporal GNN

Modular pipeline untuk prediksi tinggi muka air berbasis graph HydroRIVERS dan temporal GNN. `notebook-graph.ipynb` hanya berfungsi sebagai launcher; implementasi berada di `src/` dan entry point berada di `scripts/`.

## Struktur

```text
configs/                 Konfigurasi lokal dan Kaggle
outputs/                 Checkpoint, submission, log, experiment artifact
scripts/                 Entry point build/train/evaluate/inference/pipeline
src/data/                Loading, schema, preprocessing, split
src/features/            Feature engineering
src/graph/               Graph, adjacency, dataset, DataLoader
src/models/              Layer dan model GNN
src/training/            Engine, callbacks, checkpoint
src/evaluation/          Metrics dan evaluation
src/inference/           Autoregressive rollout dan submission
src/utils/               Config, logging, artifact, scaling
tests/                   Smoke dan unit test dasar
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r notebook-graph-modular\requirements.txt
```

## Menjalankan pipeline

Dari root workspace:

```powershell
python notebook-graph-modular/scripts/run_pipeline.py --config notebook-graph-modular/configs/default.json
```

Stage individual:

```powershell
python notebook-graph-modular/scripts/build_dataset.py --config notebook-graph-modular/configs/default.json
python notebook-graph-modular/scripts/train.py --config notebook-graph-modular/configs/default.json
python notebook-graph-modular/scripts/evaluate.py --config notebook-graph-modular/configs/default.json
python notebook-graph-modular/scripts/make_submission.py --config notebook-graph-modular/configs/default.json
```

Untuk Kaggle, gunakan `configs/kaggle.json` dan pastikan folder proyek tersedia di `/kaggle/working/notebook-graph-modular`.

## Output dan artifact

Setiap pemanggilan launcher membuat run directory di `outputs/experiments/<run-id>/`. Artifact berisi manifest environment, snapshot config, training history, evaluation metrics, dan ringkasan submission. Log per run berada di `outputs/logs/`.

## Test

```powershell
cd notebook-graph-modular
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```

Smoke test hanya membangun graph kecil dan menjalankan forward pass; tidak melatih model penuh.
