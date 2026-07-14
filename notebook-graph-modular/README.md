# Hydro Temporal GNN (modular)

Pipeline produksi modular untuk prediksi tinggi muka air berbasis HydroRIVERS dan Temporal GNN. Notebook tidak memuat ulang logika model: seluruh preprocessing, graph construction, training, evaluation, inference, dan submission berada di `src/` dan dipanggil melalui `scripts/`.

`notebook-graph-old.ipynb` di root repository dipertahankan sebagai golden reference historis. Gunakan `kaggle_launcher.ipynb` untuk Kaggle atau CLI di bawah untuk eksekusi normal.

## Entry point

Semua command menerima `--config`; `run_pipeline.py` adalah entry point utama.

```text
scripts/smoke_test.py       Import semua modul inti + graph/model forward pass CPU
scripts/build_dataset.py    Preprocessing dan build artifact NPZ
scripts/train.py            Training checkpoint baseline
scripts/evaluate.py         Evaluasi chronological validation tail
scripts/make_submission.py  Autoregressive inference + submission.csv
scripts/run_pipeline.py     Stage build/train/evaluate/submission/pipeline
```

Gunakan `--stage build|train|evaluate|submission|pipeline` dengan `run_pipeline.py`. Opsi `--dry-run` memuat konfigurasi, memvalidasi path/schema/checkpoint yang relevan, tetapi tidak membangun dataset, melatih model, atau menulis artifact.

## Menjalankan di Windows PowerShell

```powershell
git clone --branch modular-version `
  https://github.com/ibrahim119920/graph-nn-tma-mdpl.git

cd graph-nn-tma-mdpl

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\notebook-graph-modular\requirements.txt

python .\notebook-graph-modular\scripts\smoke_test.py

python .\notebook-graph-modular\scripts\run_pipeline.py `
  --config .\notebook-graph-modular\configs\default.json
```

Dataset lokal harus berada di `datasets/` pada root repository sesuai `configs/default.json`; jangan pindahkan output ke folder dataset. Untuk validasi tanpa training penuh:

```powershell
python .\notebook-graph-modular\scripts\run_pipeline.py `
  --config .\notebook-graph-modular\configs\default.json `
  --dry-run

Push-Location .\notebook-graph-modular
python -m unittest discover -s tests -v
Pop-Location
```

## Menjalankan di Kaggle

Tambahkan competition dataset ke notebook. Path input yang digunakan dan harus dipertahankan adalah:

```text
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026
/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/sample_submission.csv
```

Pilihan paling sederhana adalah upload `notebook-graph-modular/` sebagai Kaggle Dataset lalu jalankan seluruh cell pada [`kaggle_launcher.ipynb`](kaggle_launcher.ipynb). Launcher mencari project upload tersebut, menyalinnya ke lokasi writable `/kaggle/working/graph-nn-tma-mdpl/notebook-graph-modular`, atau clone branch yang benar bila project belum ada dan Internet diaktifkan.

Untuk clone eksplisit, gunakan cell berikut:

```bash
!git clone --branch modular-version \
  https://github.com/ibrahim119920/graph-nn-tma-mdpl.git \
  /kaggle/working/graph-nn-tma-mdpl

!test -d /kaggle/input/competitions/sebelas-maret-statistics-data-science-2026
!test -f /kaggle/input/competitions/sebelas-maret-statistics-data-science-2026/sample_submission.csv

!python /kaggle/working/graph-nn-tma-mdpl/notebook-graph-modular/scripts/smoke_test.py

!python /kaggle/working/graph-nn-tma-mdpl/notebook-graph-modular/scripts/run_pipeline.py \
  --config /kaggle/working/graph-nn-tma-mdpl/notebook-graph-modular/configs/kaggle.json
```

`configs/kaggle.json` selalu membaca input dari `/kaggle/input/...` dan menulis seluruh runtime output ke:

```text
/kaggle/working/graph-nn-tma-mdpl/notebook-graph-modular/outputs
```

Jangan menjalankan `pip install -r requirements.txt` di Kaggle bila environment bawaannya sudah memenuhi versi minimum; ini menghindari reinstall PyTorch/CUDA yang tidak perlu. Jika dependency geospatial belum tersedia, install requirements hanya setelah memeriksa versi preinstalled.

## Menjalankan melalui Codex

Buka root repository `graph-nn-tma-mdpl` di Codex, bukan hanya notebook atau subfolder acak. Dari root itu, gunakan command Windows di atas. Semua resolver path berpatokan pada lokasi file config, sehingga command tetap konsisten ketika current working directory berbeda.

## Struktur output

Path output dikonfigurasi di `configs/*.json`.

```text
outputs/data/          Dataset terproses (.npz)
outputs/checkpoints/   Checkpoint model terbaik
outputs/experiments/   Manifest run, config snapshot, history, metrics, ringkasan submission
outputs/logs/          Log per run
outputs/submissions/   submission.csv
```

Untuk konfigurasi lokal, folder yang sama berada di `notebook-graph-modular/outputs/`. Runtime guard menolak output yang menunjuk ke `raw_data_root`, sehingga Kaggle tidak akan mencoba menulis ke `/kaggle/input`.

## Reproducibility dan device

`runtime.seed` di config diterapkan sekali oleh entry point untuk `random`, NumPy, PyTorch CPU, dan seluruh CUDA device yang tersedia. Device dipilih aman dengan CUDA bila tersedia dan CPU bila tidak. Deterministic algorithm PyTorch tidak dipaksakan karena sebagian kernel CUDA/graph tidak selalu mendukungnya; hasil GPU dapat sedikit berbeda antar driver, CUDA, dan hardware.

## Troubleshooting

- **`ModuleNotFoundError: src`** — jalankan script melalui path `notebook-graph-modular/scripts/...` dari repository, atau buka root repository di Codex. Jangan menyalin file script sendiri tanpa folder `src/`.
- **Dataset tidak ditemukan** — periksa `raw_data_root` pada config. Kaggle harus memakai `/kaggle/input/competitions/sebelas-maret-statistics-data-science-2026`, bukan slug pendek lain.
- **CUDA tidak tersedia** — pipeline otomatis memakai CPU. Training akan lebih lambat tetapi tidak memerlukan perubahan source.
- **Checkpoint tidak cocok** — hapus hanya checkpoint yang Anda buat sendiri bila memang ingin retrain; jangan mencampur checkpoint dengan NPZ/node order/feature order/config model yang berbeda. Validasi awal menunjukkan key atau shape yang bermasalah.
- **`Permission denied`** — pastikan output berada di `outputs/` atau `/kaggle/working/.../outputs`, bukan `/kaggle/input`.
- **Submission tidak ditemukan** — lihat `outputs/logs/` dan `outputs/experiments/<run-id>/manifest.json`; command pipeline berhenti bila dataset, checkpoint, sample submission, atau submission akhir tidak valid.

## Test cepat

```powershell
python .\notebook-graph-modular\scripts\smoke_test.py
Push-Location .\notebook-graph-modular
python -m unittest discover -s tests -v
Pop-Location
python -m compileall .\notebook-graph-modular
```

Smoke test tidak membutuhkan dataset kompetisi dan tidak menjalankan training penuh.
