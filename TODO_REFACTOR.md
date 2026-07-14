# TODO Refactor `notebook-graph.ipynb`

Daftar ini adalah backlog implementasi untuk rencana pada `REFACTOR_PLAN.md`. Tidak satu pun item di bawah telah diimplementasikan oleh audit ini.

## P0 — Bekukan baseline dan cegah perubahan behavior tanpa sengaja

- [ ] Catat versi Python dan dependency geospatial/NumPy/Pandas/PyTorch.
- [ ] Tambahkan seed terpusat untuk Python, NumPy, PyTorch, CUDA, dan DataLoader.
- [ ] Buat manifest input berisi path, ukuran, timestamp/checksum, schema, rentang waktu, dan jumlah row per stasiun.
- [ ] Simpan snapshot konfigurasi notebook saat ini: periode, observation hours, lag, rolling windows, time window, graph hops, hyperparameter, dan path.
- [ ] Buat golden characterization untuk `node_order`, `feature_cols`, `edge_index`, `edge_weight`, bentuk tensor, `dt_train`, scaler, checkpoint output, dan submission.
- [ ] Tentukan toleransi numerik CPU/single-GPU/multi-GPU; jangan mengandalkan hash byte checkpoint PyTorch.
- [ ] Pisahkan baseline expected behavior dari daftar bug yang sengaja akan diperbaiki.

## P0 — Audit correctness sebelum model dipercaya

- [ ] Konfirmasi provenance dan availability setiap kolom environment pada waktu inference.
- [ ] Verifikasi definisi `rainfall_max_24h_mm` adalah trailing window, bukan forward-looking window.
- [ ] Ukur gap environment dan water level per stasiun; tetapkan maksimum forward-fill yang diperbolehkan.
- [ ] Perbaiki desain validasi rollout agar tidak menyalin aktual validation setelah gap secara implisit.
- [ ] Bila warm-start per segmen memang dibutuhkan, beri nama metric terpisah dan laporkan jumlah/length segmen.
- [ ] Buat full-horizon causal rollout metric yang sama dengan inference test.
- [ ] Audit preprocessing skew: training membuang window NaN sementara inference meng-forward-fill 4.884 sel `wl_t` pada periode train.
- [ ] Tentukan kebijakan konsisten untuk missing water level pada training, validation, dan inference.
- [ ] Verifikasi graph benar-benar connected; jangan menyimpulkan connected hanya karena tidak ada isolated node.
- [ ] Deteksi beberapa stasiun yang snap ke `HYRIV_ID` sama dan tentukan aturan collision.

## P1 — Buat skeleton package dan konfigurasi

- [ ] Buat package `src/hydro_gnn/` dan scripts sesuai struktur pada `REFACTOR_PLAN.md`.
- [ ] Buat `PathConfig`, `DataConfig`, `FeatureConfig`, `GraphConfig`, `ModelConfig`, `TrainingConfig`, dan `InferenceConfig`.
- [ ] Jadikan konfigurasi satu-satunya sumber nilai periode, lag, rolling, window, arsitektur, dan path.
- [ ] Tambahkan resolver path lokal/Kaggle tanpa edit source.
- [ ] Simpan config serialization dan config hash pada dataset artifact/checkpoint/submission manifest.
- [ ] Tetapkan feature schema version dan artifact version.

## P1 — Ekstrak input dan validasi data

- [ ] Ekstrak `load_hydrorivers()`.
- [ ] Ekstrak `load_station_coordinates()`.
- [ ] Ekstrak `load_train_data()`.
- [ ] Ekstrak `load_environment_data()`.
- [ ] Validasi kolom wajib dan dtype.
- [ ] Validasi timezone/naive datetime policy dan parsing failure.
- [ ] Validasi uniqueness `(datetime, nama_pos)` pada train/environment.
- [ ] Validasi uniqueness `nama_pos` pada koordinat.
- [ ] Validasi latitude/longitude finite dan berada pada range yang benar.
- [ ] Validasi station set antar-source; ubah warning menjadi error atau policy eksplisit.
- [ ] Buat coverage report per stasiun/periode/jam observasi.
- [ ] Uji bahwa target tidak tersedia pada periode test yang akan diprediksi.

## P1 — Ekstrak graph construction

- [ ] Ekstrak `snap_stations_to_rivers()` tanpa mengubah hasil baseline terlebih dahulu.
- [ ] Ekstrak `trace_downstream_station()` dengan dependency map sebagai argumen.
- [ ] Ekstrak `build_river_graph()`.
- [ ] Ekstrak `extract_river_features()`.
- [ ] Tambahkan test cycle dan `MAX_DOWNSTREAM_HOPS`.
- [ ] Deteksi nearest-snap tie dan buat tie-break deterministic.
- [ ] Deteksi duplicate snapped `HYRIV_ID`.
- [ ] Validasi edge index range, self-edge, duplicate edge, symmetry, dan finite/nonnegative weight.
- [ ] Hitung connected components dan degree distribution setelah fallback.
- [ ] Rancang fallback yang menyatukan komponen, bukan hanya node berderajat nol.
- [ ] Dokumentasikan bahwa edge distance saat ini bukan jarak station-to-station presisi.
- [ ] Evaluasi CRS lokal yang lebih tepat untuk nearest distance; lakukan sebagai perubahan behavior terpisah.
- [ ] Hapus `connected_pairs` setelah dipastikan tidak diperlukan.

## P1 — Ekstrak preprocessing dan feature engineering

- [ ] Ekstrak `build_observation_index()`.
- [ ] Ekstrak `prepare_environment_features()`.
- [ ] Ekstrak `build_water_level_features()`.
- [ ] Ekstrak `build_time_features()`.
- [ ] Ekstrak `build_spatial_features()`.
- [ ] Ekstrak `assemble_node_features()` dan `build_next_step_target()`.
- [ ] Pastikan semua fungsi menerima input/config dan tidak membaca global notebook.
- [ ] Tambahkan causality tests untuk lag, rolling, dan imputasi.
- [ ] Tambahkan test leap year dan pola interval 06:00–12:00–18:00–06:00.
- [ ] Tambahkan opsi/fitur `delta_time` bila interval tak seragam dipertahankan.
- [ ] Encode `wind_direction_deg` secara siklik setelah characterization baseline.
- [ ] Tentukan encoding `mjo_phase` dan `landcover_class` sebagai kategori/siklik.
- [ ] Keluarkan atau encode `MAIN_RIV`/`HYBAS_L12` sebagai identifier, bukan kontinu.
- [ ] Tambahkan missing indicator dan age-since-observation bila terbukti berguna.
- [ ] Batasi forward-fill berdasarkan durasi maksimum dan laporkan nilai yang melewati batas.

## P1 — Ekstrak tensor dataset dan artifact

- [ ] Ekstrak `find_valid_timesteps()`.
- [ ] Ekstrak `build_feature_panel()`.
- [ ] Ekstrak `build_supervised_windows()`.
- [ ] Assertion `dt_train` strictly increasing dan target tepat satu step setelah endpoint.
- [ ] Validasi spacing/window di sekitar gap.
- [ ] Buat `DatasetArtifact` dengan metadata config, schema, node order, dtype, shapes, dan fingerprint.
- [ ] Hindari object arrays/`allow_pickle=True` bila memungkinkan; simpan metadata sebagai JSON aman.
- [ ] Pisahkan graph artifact, supervised dataset, dan inference panel jika memberi lifecycle yang lebih jelas.
- [ ] Putuskan satu sumber target test: sample submission atau `test_valid_timesteps`; hapus state yang tidak digunakan.
- [ ] Hapus global `output_path` yang tidak digunakan atau jadikan path canonical.

## P1 — Ekstrak model, scaling, dan checkpoint

- [ ] Pindahkan `GCNLayer` ke `models/layers.py`.
- [ ] Pindahkan baseline `TemporalGNN` ke `models/temporal_gnn.py`.
- [ ] Pindahkan residual model ke `models/residual_temporal_gnn.py`.
- [ ] Hapus definisi model duplikat pada inference.
- [ ] Buat scaler serializable dengan `fit/transform/inverse_transform` dan schema check.
- [ ] Pastikan scaler hanya fit pada split train.
- [ ] Simpan architecture name dan seluruh constructor args di checkpoint.
- [ ] Simpan feature schema, node order, graph fingerprint, config hash, metric, dan training datetime di checkpoint.
- [ ] Buat satu loader checkpoint yang memvalidasi kompatibilitas dataset/model.
- [ ] Samakan contract checkpoint baseline dan residual.
- [ ] Tambahkan test adjacency normalization dan output shape model.

## P1 — Ekstrak training dan validation engine

- [ ] Buat `GraphTimeSeriesDataset` reusable.
- [ ] Buat `train_epoch(model, loader, optimizer, criterion, device)` tanpa lookup global.
- [ ] Buat `evaluate_epoch(model, loader, criterion, device)`.
- [ ] Timbang mean loss berdasarkan jumlah sample, bukan jumlah batch.
- [ ] Buat trainer dengan early stopping, scheduler, gradient clipping config, dan structured history.
- [ ] Pilih state dict berdasarkan `isinstance(model, nn.DataParallel)`, bukan flag terpisah.
- [ ] Dukung CPU/single-GPU/multi-GPU; hilangkan hard error dua GPU pada residual experiment.
- [ ] Tangani rollout NaN/`inf` dan kasus tidak ada best state dengan error terdiagnosis.
- [ ] Pisahkan teacher-forced metrics dari rollout metrics.
- [ ] Laporkan jumlah sample/timestep/segmen yang masuk tiap metric.
- [ ] Pertimbangkan purge gap train/validation dan dokumentasikan trade-off.
- [ ] Tentukan apakah final model dilatih ulang pada seluruh train setelah model selection.

## P1 — Satukan causal rollout dan inference

- [ ] Buat satu `rebuild_water_window()` untuk validation dan inference.
- [ ] Buat stateful causal rollout engine yang tidak membaca actual future secara tersembunyi.
- [ ] Gunakan metadata `N_LAGS`, rolling windows, dan `TIME_WINDOW` dari artifact/checkpoint, bukan hard-coded.
- [ ] Validasi urutan fitur/node/dimensi sebelum prediksi.
- [ ] Tangani target timestamp nonkontigu secara eksplisit.
- [ ] Klasifikasikan setiap skipped target berdasarkan alasan.
- [ ] Dokumentasikan dan uji clipping guardrail.
- [ ] Pisahkan raw model prediction dan postprocessed prediction pada metric/artifact.
- [ ] Jadikan pemilihan baseline/residual eksplisit melalui config/checkpoint path.

## P1 — Submission

- [ ] Parse ID dengan delimiter yang tervalidasi, bukan indeks karakter tetap.
- [ ] Validasi ID unik pada sample submission.
- [ ] Validasi kesetaraan set `(datetime, nama_pos)` target dan prediksi.
- [ ] Validasi order output identik dengan sample submission.
- [ ] Validasi nama kolom persis, row count, dtype numeric, finite, dan no-NaN.
- [ ] Ganti fallback diam-diam dengan policy eksplisit dan log per baris/stasiun.
- [ ] Tangani stasiun tanpa satu pun prediksi sebelum mean fallback.
- [ ] Simpan submission manifest: checkpoint, dataset fingerprint, config, metric, dan timestamp.

## P2 — Visualisasi dan diagnostics

- [ ] Ekstrak plot learning history.
- [ ] Ekstrak plot baseline-vs-residual.
- [ ] Ekstrak plot per-station actual-vs-prediction.
- [ ] Pilih stasiun diagnostic secara data-driven (gap/error/drift terbesar), bukan list manual saja.
- [ ] Tambahkan missingness heatmap dan gap-duration plot.
- [ ] Tambahkan graph topology/connected-component/degree plot.
- [ ] Tambahkan residual-over-time dan actual-vs-predicted scatter.
- [ ] Simpan semua figure dengan path/config yang konsisten dan tutup figure setelah save/show.
- [ ] Pisahkan load/transform data dari fungsi plotting.

## P2 — Bersihkan bagian sekali pakai setelah parity tercapai

- [ ] Pindahkan sample prints cell 3/8 ke diagnostic command atau hapus dari production path.
- [ ] Pindahkan debug cell 30/31 ke validator/test otomatis.
- [ ] Jadikan cell 34 script eksperimen terpisah agar tidak menimpa baseline state.
- [ ] Pindahkan quick plots cell 37/39 ke scripts/diagnostics.
- [ ] Hapus import tidak digunakan (`Point`, `math`, dan import duplikat lain) setelah ekstraksi.
- [ ] Hapus variabel tidak digunakan (`connected_pairs`, `output_path`) setelah characterization.

## P2 — Tests dan acceptance

- [ ] Unit test schema dan station ordering.
- [ ] Unit test causal environment fill tanpa akses future.
- [ ] Unit test lag/rolling dengan seri kecil yang hasilnya dihitung manual.
- [ ] Unit test target shift pada batas hari dan batas train/test.
- [ ] Unit test graph traversal cycle/collision/disconnected components.
- [ ] Unit test window builder pada gap dan awal seri.
- [ ] Unit test scaler tidak fit pada validation.
- [ ] Unit test model save/load parity.
- [ ] Unit test validation rollout tidak mengakses actual future.
- [ ] Unit test inference untuk target kontigu dan nonkontigu.
- [ ] Unit test submission parsing/order/duplicate/missing/finite.
- [ ] Integration test `build_dataset → train → predict → validate_submission` dari proses bersih.
- [ ] Bandingkan baseline modular dengan notebook pada golden characterization.
- [ ] Setelah parity, ukur ulang metric untuk setiap bug fix secara terpisah.

## Definition of done

- [ ] Seluruh mapping Cell → fungsi → file pada `REFACTOR_PLAN.md` telah diwujudkan atau diberi alasan tertulis bila dikecualikan.
- [ ] Tidak ada fungsi production yang bergantung pada global notebook state.
- [ ] Config, feature schema, node order, dan artifact/checkpoint compatibility tervalidasi otomatis.
- [ ] Validation rollout dan inference berbagi causal engine yang sama.
- [ ] Pipeline berjalan dari proses bersih pada environment lokal dan Kaggle.
- [ ] Baseline parity dan hasil bug-fix terdokumentasi terpisah.
- [ ] Notebook akhir, bila dipertahankan, hanya memanggil API package dan menampilkan laporan.
