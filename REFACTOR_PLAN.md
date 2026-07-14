# Historical Refactor Plan — `notebook-graph.ipynb`

> **Status aktual (2026-07-14):** rencana ini adalah audit sebelum modularisasi. Implementasi modular sekarang berada di `notebook-graph-modular/` pada branch `modular-version`; notebook lama dipertahankan sebagai golden reference. Deskripsi “belum dibuat” di dokumen ini bersifat historis, bukan status repository saat ini. Lihat `notebook-graph-modular/README.md` untuk cara menjalankan project.

## 1. Ruang lingkup audit

Dokumen ini adalah hasil audit statis atas seluruh 40 cell pada `notebook-graph.ipynb` (cell 0–39), termasuk source, execution count, dan output yang tersimpan. Dokumen ini hanya merencanakan ekstraksi notebook ke modul Python; tidak ada refactor atau perubahan perilaku yang dilakukan.

Notebook yang tersimpan menunjukkan eksekusi berurutan untuk seluruh code cell (`1`–`23`). Karena itu, output saat ini konsisten dengan skenario **Run All dari kernel bersih**, tetapi konsistensi tersebut belum dijamin jika cell dijalankan ulang secara parsial atau tidak berurutan.

Ringkasan artefak pada output notebook:

- 30 node/stasiun.
- 66 fitur per node per timestep.
- Graph berisi 27 edge sebelum dibuat dua arah, atau 54 edge terarah pada `edge_index`.
- 819 sample train dengan bentuk `(819, 12, 30, 66)` dan target `(819, 30)`.
- Split tersimpan: 738 sample train dan 81 sample validation.
- Horizon submission: 726 timestep × 30 stasiun = 21.780 baris.
- Baseline terbaik: rollout normalized RMSE 0,35839; teacher-forced RMSE 0,2666 dan MAE 0,1560.
- Eksperimen residual: teacher-forced RMSE 0,2047, tetapi rollout normalized RMSE 0,3730 (lebih buruk daripada baseline); inference akhirnya memakai checkpoint baseline.

## 2. Struktur notebook saat ini

Alur notebook terbagi secara implisit menjadi blok berikut:

1. **Cell 0–8 — input dan inspeksi awal**: mengatur path Kaggle, membaca HydroRIVERS, koordinat, train, dan data lingkungan.
2. **Cell 9–14 — dokumentasi data dan konfigurasi**: mendokumentasikan schema secara manual, menetapkan periode, lag, rolling window, dan urutan node.
3. **Cell 15–16 — graph construction**: snap stasiun ke sungai, menelusuri `NEXT_DOWN`, memberi fallback spasial, membentuk `edge_index`, `edge_weight`, dan atribut sungai statis.
4. **Cell 17–24 — preprocessing dan feature engineering**: membentuk grid observasi, imputasi lingkungan, lag/rolling tinggi air, fitur waktu/spasial, dan tabel fitur long-format.
5. **Cell 25–28 — dataset tensor**: memeriksa kelengkapan timestep, membentuk tensor train, dan menyimpan bahan inference ke NPZ.
6. **Cell 29–31 — debugging satu kali**: mengecek missingness pada test dan memastikan `train.csv` tidak memiliki baris pada periode test.
7. **Cell 32–33 — baseline training**: load NPZ, split waktu, fit scaler, definisi model, training, rollout validation, checkpointing, dan evaluasi.
8. **Cell 34 — eksperimen residual**: mengganti global training state, melatih model residual, membandingkan metrik, menyimpan checkpoint dan plot.
9. **Cell 35–37 — inference dan inspeksi cepat**: mendefinisikan ulang model baseline, autoregressive rollout, submission, lalu plot tiga stasiun.
10. **Cell 38–39 — final visual check**: membandingkan train aktual dan submission untuk seluruh stasiun.

Masalah struktural utamanya bukan ukuran notebook semata, melainkan satu cell training yang memuat terlalu banyak tanggung jawab (cell 33, 403 baris) dan satu cell inference yang juga monolitik (cell 36, 261 baris). Batas antara konfigurasi, transformasi data, model, evaluasi, dan efek samping file belum eksplisit.

## 3. Dependency antar-cell dan urutan eksekusi

### 3.1 Dependency utama

```text
Cell 1 (path)
├─> 2 (gdf) ─> 3 (inspeksi)
├─> 4 (coordinate)
├─> 6 (train_data)
└─> 7 (environment_data) ─> 8 (inspeksi)

Cell 4 + 6 + 7 + 12
└─> 14 (node_order, coordinate_idx)

Cell 2 + 12 + 14
└─> 16 (graph, edge, river_attrs)

Cell 6 + 7 + 12 + 14
└─> 18 (obs grid, train_obs, env_full)
    ├─> 20 (fitur air)
    └─> 22 (fitur waktu; juga memakai 12)

Cell 16 + 18 + 20 + 22
└─> 24 (node_features, feature_cols, y_next)
    ├─> 26 (tensor dataset)
    │   └─> 28 (NPZ)
    └─> 30–31 (debug)

NPZ dari 28
└─> 33 (baseline train + validation + checkpoint)
    ├─> 34 (eksperimen residual; memakai dan menimpa banyak global 33)
    └─> 36 (inference baseline; load checkpoint/NPZ dari disk)
        ├─> 37 (plot pred_df)
        └─> 39 (load OUTPUT_PATH dan plot final; juga memakai path/plt/np/pd global)
```

### 3.2 Urutan aman saat ini

Urutan aman adalah code cell `1, 2, 3, 4, 6, 7, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 34, 36, 37, 39`. Cell markdown dapat dilewati oleh runtime.

Urutan minimum per produk:

- **Dataset NPZ**: 1 → 2 → 4 → 6 → 7 → 12 → 14 → 16 → 18 → 20 → 22 → 24 → 26 → 28.
- **Baseline checkpoint**: dataset NPZ tersedia → 33.
- **Residual checkpoint/plot**: baseline state cell 33 tersedia → 34.
- **Submission baseline**: dataset NPZ + baseline checkpoint + sample submission tersedia → 36.
- **Final check**: cell 36 atau file submission pada `OUTPUT_PATH` tersedia; juga perlu import yang saat ini diwarisi dari cell sebelumnya → 39.

### 3.3 Hidden notebook state

Risiko state tersembunyi yang konkret:

- Cell 33 dan 34 menggunakan nama global yang sama untuk `model`, `criterion`, `optimizer`, dan `scheduler`. Setelah cell 34, pemanggilan ulang `run_epoch` atau `run_rollout_validation` dari cell 33 akan memakai model/optimizer residual, bukan baseline.
- `run_rollout_validation()` membaca `model` dari global saat fungsi dipanggil. Perilakunya berubah tanpa perubahan fungsi ketika cell 34 menimpa `model`.
- `run_epoch()` juga menutup (closure via global lookup) `model`, `optimizer`, dan `criterion`; fungsi bukan unit baseline yang stabil.
- `GCNLayer` dan `TemporalGNN` didefinisikan ulang pada cell 36. Instance/model yang dibuat sebelum dan sesudah redefinisi dapat berasal dari class object berbeda walau namanya sama.
- Cell 39 bergantung pada `pd`, `np`, dan `plt` dari cell lama, bukan mengimpor semuanya sendiri. Ia hanya mengimpor `Path`, `math`, dan `matplotlib.dates`; `math` justru tidak dipakai.
- Cell 36 mengulang dan menimpa `N_LAGS`, `TIME_WINDOW`, `TRAIN_END`, `CKPT_PATH`, dan `DATA_PATH` tanpa pemeriksaan terhadap konfigurasi pembuatan dataset.
- Cell 28 membuat `output_path`, tetapi pemanggilan `np.savez_compressed` membangun path lain secara inline; global `output_path` tidak menjadi sumber kebenaran.
- Cell 34 bergantung pada `rmse`, `mae`, `per_node_rmse`, `best_val_loss`, loader, scaler, graph, dan helper dari cell 33. Menjalankannya langsung dari kernel bersih pasti gagal.
- Cell 37 bergantung pada `pred_df` dari cell 36.
- Cell 39 bergantung pada `path` dari cell 1 dan `OUTPUT_PATH` dari cell 36.
- Semua execution count tersimpan memang berurutan, tetapi tidak ada mekanisme runtime yang menolak notebook state yang stale setelah satu cell hulu diubah lalu cell hilir tidak dijalankan ulang.

## 4. Inventaris variabel global

### 4.1 Konfigurasi dan path

`path`, `OBS_HOURS`, `TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END`, `N_LAGS`, `ROLLING_WINDOWS`, `TIME_WINDOW`, `MAX_DOWNSTREAM_HOPS`, `OUTPUT_DIR`, `output_path`, `DATA_PATH`, `CKPT_PATH`, `SAMPLE_SUB_PATH`, `OUTPUT_PATH`, `BATCH_SIZE`, `EPOCHS`, `LR`, `PATIENCE`, `USE_MULTI_GPU`, `EPOCHS_IMPROVED`, `IMPROVED_CKPT`, `GRAPH_PATH`.

Audit:

- Konfigurasi tersebar di cell 1, 12, 28, 33, 34, dan 36.
- Nilai yang sama ditulis ulang secara manual (`N_LAGS`, `TIME_WINDOW`, `TRAIN_END`).
- Path hanya cocok untuk Kaggle saat ini; alternatif lokal dikomentari.
- Arsitektur baseline pada load checkpoint di-hard-code lagi (`64/64`), meskipun metadata arsitektur tidak lengkap disimpan sebagai kontrak model.
- Tidak ada seed, versi dataset, versi feature schema, atau fingerprint konfigurasi.

### 4.2 DataFrame/array pipeline

`gdf`, `coordinate`, `train_data`, `environment_data`, `node_order`, `node_to_idx`, `coordinate_idx`, `snapped`, `river_attrs`, `train_obs`, `env_full`, `wl_long`, `time_feat`, `spatial_feat`, `node_features`, `feature_cols`, `panel_arr`, `target_arr`, `X_train`, `y_train`, `dt_train`, `test_valid_timesteps`, `edge_index`, `edge_weight`.

Audit:

- Objek besar hidup bersamaan di kernel dan sebagian disalin berkali-kali.
- Nama seperti `data`, `panel_arr`, dan `model` digunakan ulang pada fase berbeda.
- Kontrak bentuk, dtype, urutan node, dan urutan fitur hanya tersirat oleh code dan print.
- `allow_pickle=True` diperlukan karena NPZ menyimpan object array (`feature_cols`, `node_order`), mengurangi portabilitas dan menambah risiko saat memuat artefak yang tidak tepercaya.

### 4.3 State training/inference

`device`, `model`, `criterion`, `optimizer`, `scheduler`, `train_loader`, `val_loader`, scaler arrays, adjacency, history, checkpoint metrics, `wl_series`, `predictions`, `pred_df`, `submission_final`.

Audit:

- Training engine tidak menerima dependency sebagai argumen.
- Scaler tersimpan sebagai sekumpulan array tanpa objek/schema version.
- Model selection baseline vs residual tidak dinyatakan sebagai konfigurasi eksplisit; cell inference diam-diam memilih baseline.

## 5. Audit per area

### 5.1 Dataset dan input

Hal yang sudah baik:

- Ada urutan node kanonik dari koordinat.
- Pos yang hilang dari train/environment dibandingkan terhadap koordinat.
- Grid `(datetime, nama_pos)` eksplisit dibuat sebelum feature engineering.
- Output notebook mencatat 30 stasiun dan tidak menampilkan missing station set.

Risiko/kekurangan:

- Tidak ada validasi schema, tipe, timezone, duplikasi key `(datetime, nama_pos)`, nilai koordinat, urutan waktu, atau frekuensi aktual.
- `set_index(...).reindex(...)` akan gagal pada duplicate key, tetapi pesan error tidak dikontekstualkan.
- Duplikasi `nama_pos` pada koordinat dapat membuat `.set_index().loc[node_order]` menghasilkan lebih dari satu baris per node.
- Dokumentasi schema pada cell 9 manual dan dapat drift dari file aktual.
- Periode train yang dikonfigurasi lebih panjang daripada kepadatan observasi aktual beberapa stasiun; output final memperlihatkan jumlah row per stasiun tidak seragam (misalnya Gunungsari 1.126 versus banyak stasiun sekitar 1.426–1.428), tetapi tidak ada laporan coverage formal.
- Tidak ada checksum/fingerprint input, sehingga checkpoint dan dataset bisa tercampur dengan versi input berbeda.

### 5.2 Preprocessing

Hal yang sudah baik:

- Imputasi lingkungan memakai forward-fill per stasiun.
- Fallback mean dihitung hanya dari periode train.
- Pemisahan scaler dilakukan sebelum fitting; scaler hanya memakai bagian training dari split.

Risiko/kekurangan:

- Forward-fill lingkungan tidak memiliki batas maksimum gap. Nilai lama dapat dipakai berbulan-bulan tanpa flag umur observasi.
- Mean fallback global per kolom, bukan per stasiun; ini mungkin tepat atau tidak, tetapi keputusan tidak diuji.
- Tidak ada missing indicator atau `time_since_last_observation`.
- Inference melakukan forward-fill `wl_t` periode train sebanyak 4.884 sel, sementara training membuang window yang mengandung NaN. Ini menciptakan train–inference preprocessing skew yang besar.
- Leading NaN tinggi air tidak diisi; assert hanya memeriksa histori yang diperlukan pada batas train/test, bukan seluruh seri.
- Fitur statis dan identifier ikut standardisasi generik tanpa klasifikasi semantik.

### 5.3 Feature engineering

Hal yang sudah baik:

- Lag memakai `shift` per stasiun.
- Rolling memakai seri yang sudah `shift(1)`, sehingga tidak memasukkan `wl_t` ke statistik histori.
- Target adalah nilai langkah berikutnya per stasiun.
- Fitur waktu tahunan mempertimbangkan leap year.

Risiko/kekurangan:

- `wind_direction_deg` digunakan sebagai angka linear; 359° dan 1° dianggap jauh. Seharusnya direncanakan encoding siklik.
- `mjo_phase` dan `landcover_class` adalah kategori/kode tetapi diperlakukan kontinu. `MAIN_RIV` dan `HYBAS_L12` merupakan identifier, bukan besaran numerik; float32 juga tidak aman untuk mempertahankan integer ID besar secara presisi.
- `rmm1/rmm2` sudah merupakan koordinat kontinu; `mjo_phase` berpotensi redundan dan perlu keputusan eksplisit.
- Arti `rainfall_max_24h_mm` belum diverifikasi: bila dibentuk dari window ke depan, fitur ini leak; bila trailing, aman. Provenance fitur lingkungan perlu didokumentasikan.
- Ketersediaan fitur lingkungan test saat inference produksi/kompetisi belum diformalkan. Data cuaca aktual di masa target dapat menjadi leakage operasional jika pada skenario nyata hanya forecast yang tersedia.
- `feature_cols` dibangun dari nama kolom dinamis. Tidak ada schema version atau assertion urutan terhadap checkpoint.
- `pd.date_range(..., freq="6h")` lalu filter jam menghasilkan jeda 6 jam, 6 jam, dan 12 jam. Model memperlakukan setiap step setara; ini mungkin sesuai tiga observasi/hari, tetapi interval yang tidak seragam perlu dinyatakan sebagai asumsi atau ditambah fitur `delta_time`.

### 5.4 Graph construction

Hal yang sudah baik:

- Snap dilakukan dalam CRS metrik.
- Traversal memiliki visited set dan batas hop.
- Node benar-benar terisolasi diberi fallback nearest spatial neighbor.
- Graph dibuat dua arah dan adjacency diberi self-loop.

Risiko/bug potensial:

- Klaim “agar graph tetap terhubung” belum dibuktikan. Menyambungkan node berderajat nol tidak menjamin beberapa komponen non-isolated menjadi satu connected component; tidak ada connected-components assertion.
- `hyriv_to_pos = {v: k ...}` menghilangkan stasiun bila dua stasiun snap ke `HYRIV_ID` yang sama. Collision tidak diperiksa.
- Tie pada `sjoin_nearest` lalu `drop_duplicates(..., keep="first")` dapat memilih segmen secara tidak deterministik.
- Node dengan koordinat NaN tidak ditolak di awal. Ia hilang dari `coord_clean`, lalu dapat menghasilkan atribut sungai NaN atau fallback distance NaN.
- Edge distance menjumlahkan panjang penuh segmen hilir, bukan jarak aktual antara titik stasiun di sepanjang sungai; nama `weight_km` berpotensi memberi presisi semu.
- `connected_pairs` dibuat tetapi tidak digunakan.
- Fallback hanya untuk node tanpa edge, bukan untuk menyatukan komponen graph.
- Tidak ada audit self-edge, duplicate edge, negative/zero/nonfinite weight, symmetry, degree distribution, atau isolated node setelah fallback.
- EPSG:3857 cukup praktis untuk nearest search, tetapi distorsi jarak lokal dan pilihan CRS belum dicatat.
- Atribut sungai ID/kategori dimasukkan sebagai fitur numerik mentah.

### 5.5 Tensor dataset

Hal yang sudah baik:

- Reindex eksplisit menjamin urutan waktu dan node.
- Window yang tidak lengkap ditolak.
- Waktu sample train tersimpan sebagai `dt_train`.

Risiko/kekurangan:

- `test_valid_timesteps` disimpan tetapi tidak digunakan oleh inference; sumber target justru sample submission.
- `wl_col_positions` disimpan, namun inference menghitung ulang beberapa posisi dari string dan hard-coded `N_LAGS`.
- Tidak ada metadata konfigurasi lengkap, versi feature schema, dtype contract, atau validasi saat load.
- Seluruh `panel_arr` train+test disimpan bersama. Ini sah untuk fitur exogenous yang memang tersedia, tetapi provenance dan availability contract harus eksplisit.
- NPZ tunggal mencampur train tensor, graph, schema, dan panel inference; perubahan salah satu komponen memaksa regenerasi seluruh artefak.
- Tidak ada assertion bahwa `dt_train` strictly increasing, bahwa semua window memiliki spacing yang diharapkan, atau bahwa target tepat satu langkah setelah endpoint window.

### 5.6 Model

Baseline adalah dua layer GCN dense per timestep, local linear skip, GRU per node, dan MLP head. Desain sesuai graph kecil (30 node) dan bentuk tensor saat ini.

Risiko/kekurangan:

- Arsitektur, dimensi, dan dropout tersebar antara constructor, instantiation, dan loader inference.
- Definisi model diduplikasi pada training dan inference.
- Residual model tidak memiliki loader inference; checkpoint-nya juga tidak memuat semua metadata yang sama dengan baseline.
- Dense adjacency sesuai untuk 30 node, tetapi tidak ada test numerik normalization/symmetry.
- Edge weight `1/(km+1e-3)` memberi skala yang dapat sangat dominan untuk edge pendek; tidak ada clipping/transform comparison.
- Feature identifier/kategori masuk ke linear layer sebagai nilai kontinu.

### 5.7 Training loop

Hal yang sudah baik:

- Split kronologis dilakukan sebelum fit scaler.
- Huber loss, weight decay, scheduler, checkpoint, dan early stopping tersedia.
- Checkpoint dipilih berdasarkan rollout metric, bukan hanya teacher-forced loss.
- Ada guardrail per-station dari quantile training.

Risiko/bug potensial:

- Tidak ada random seed untuk Python/NumPy/PyTorch/DataLoader/CUDA; hasil tidak reproducible.
- Loss epoch dirata-ratakan per batch, bukan dibobot berdasarkan jumlah sample; batch terakhir memiliki bobot yang sama dengan batch penuh.
- Cell baseline memegang data prep, model, training, validation, checkpoint, dan reporting sekaligus.
- Final model tidak dilatih ulang pada seluruh data train setelah model selection; 10% data terakhir tidak dipakai untuk fit checkpoint final.
- Tidak ada gradient clipping pada baseline, logging terstruktur, config snapshot, atau resume contract.
- `USE_MULTI_GPU` dihitung dari jumlah GPU, sedangkan state dict selection mengandalkan nilai itu alih-alih memeriksa tipe model secara langsung.
- Eksperimen residual hard-fail jika GPU kurang dari dua, meskipun model secara teknis dapat dijalankan pada satu GPU/CPU.
- `best_improved_state` dapat tetap `None` jika semua rollout metric `inf`/NaN, lalu `load_state_dict(None)` gagal tanpa diagnostic yang jelas.

### 5.8 Validation

Temuan paling kritis ada pada restart segmen di `run_rollout_validation()`:

```python
if np.isnan(rollout_wl[input_pos]).any():
    rollout_wl[:input_pos + 1] = actual_wl_all[:input_pos + 1]
```

Saat valid timestep terputus, baris ini mengembalikan seluruh nilai aktual sampai `input_pos`, termasuk nilai di dalam rentang validation. Ini membuat rollout setelah gap tidak lagi murni autoregresif dan dapat memberi metric optimistis. Perilaku tersebut mungkin dimaksudkan sebagai evaluasi per-segmen dengan warm start aktual, tetapi harus disebut sebagai metric berbeda dan tidak boleh disamakan dengan long-horizon test rollout.

Risiko lain:

- Satu validation tail dipakai untuk early stopping, scheduler, pemilihan baseline, dan perbandingan eksperimen residual; tidak ada test/holdout kedua untuk estimasi final.
- Teacher-forced dan rollout metric mengukur rezim berbeda, tetapi nama/reporting belum menyertakan jumlah sample/segmen yang benar-benar dinilai.
- Rollout metric dinormalisasi per stasiun, sedangkan teacher-forced RMSE dilaporkan pada skala asli secara global; perbandingan lintas metric mudah disalahartikan.
- Guardrail clipping ikut menentukan validation rollout. Ini sah jika juga bagian inference, tetapi metric model dan metric sistem pasca-proses perlu dibedakan.
- Sample train berdekatan memiliki window yang overlap. Ini normal untuk time series, tetapi gap/purge antara train dan validation dapat dipertimbangkan untuk evaluasi yang lebih konservatif.

### 5.9 Inference

Hal yang sudah baik:

- Target time berasal dari sample submission.
- Window berakhir satu langkah sebelum target.
- Lag dan rolling dibangun ulang dari gabungan aktual/prediksi secara kausal.
- Tidak memakai backfill.
- Ada assert histori batas train/test, clipping, shape check, dan no-NaN check.

Risiko/bug potensial:

- Inference memilih checkpoint baseline walaupun cell residual dijalankan; keputusan model tidak eksplisit.
- Konfigurasi lag/window/periode diulang dan bisa drift dari dataset/checkpoint. `ckpt["time_window"]` dimuat tetapi nilai hard-coded tetap dipakai.
- Tidak ada assertion bahwa `feature_cols`, `node_order`, `num_nodes`, `num_features`, dan graph fingerprint pada NPZ identik dengan checkpoint.
- Forward-fill tinggi air inference berbeda dari kebijakan training.
- Jika target timestamp tidak kontigu, prediksi timestamp berikutnya dapat gagal karena `wl_series` pada step yang dilewati tetap NaN.
- Fallback submission melakukan ffill lalu mean dari prediksi yang tersedia. Jika satu stasiun sama sekali tidak memiliki prediksi, mean tetap NaN; assert baru gagal di akhir.
- Parsing ID menggunakan slicing posisi tetap (`[:19]`, `[22:]`), rapuh terhadap perubahan format/spasi.
- `skipped` mencampur penyebab berbeda (di luar range, kurang history, fitur NaN) tanpa diagnostic per alasan.
- Tidak ada validasi range/finite/outlier setelah submission selain clipping internal.

### 5.10 Submission

- Jumlah baris dan NaN diperiksa; order direkonstruksi melalui left merge dengan `sample_sub`, sehingga secara praktik mengikuti urutan sample.
- Belum ada assertion ID unik, kesetaraan set ID, nama kolom persis, dtype numeric finite, atau duplikasi setelah merge.
- Fallback prediksi tidak dicatat ke artefak audit per baris.
- Nama/model/checkpoint/config hash tidak disimpan bersama submission.

### 5.11 Visualisasi

- Cell 34 memberi learning curves, rollout curve, comparison metric, dan per-station gain; plot disimpan.
- Cell 37 memberi smoke plot tiga stasiun yang dipilih manual.
- Cell 39 memplot semua 30 stasiun dalam lima gambar.

Kekurangan:

- Pemilihan tiga stasiun tidak otomatis berdasarkan gap/error terbesar.
- Plot final tidak menyimpan file dan tidak memberi band/threshold atau menandai gap data.
- Tidak ada visual graph/topology, degree/connected-component, missingness heatmap, residual time plot, atau actual-vs-pred scatter.
- Cell visualisasi masih melakukan load/transform data sendiri, bukan menerima data siap plot.

### 5.12 Bagian sekali pakai dan reusable

Bagian yang hanya digunakan sekali atau bersifat eksploratif:

- Cell 3, 8: sample print data.
- Cell 30, 31: debugging missing test dan konfirmasi test target kosong.
- Cell 34: eksperimen residual satu kali dan comparison plot.
- Cell 37: quick plot tiga stasiun.
- Cell 39: final manual visual check.
- `connected_pairs` (cell 16), `output_path` (cell 28), dan import `Point`/`math` tidak dipakai.

Bagian yang layak reusable:

- Loader dan validator schema data.
- Pembentuk observation grid.
- Imputasi causal environment/water level.
- Lag/rolling/time feature builders.
- Snap/traversal/fallback/graph validation.
- Tensor window builder.
- Scaler per stasiun dan feature scaler.
- `GCNLayer`, baseline model, residual model.
- Training epoch, rollout engine, metric aggregation, checkpoint I/O.
- Autoregressive water-feature rebuild.
- Submission formatter/validator.
- Plotting diagnostics.

## 6. Kemungkinan data leakage

| Risiko | Status audit | Tingkat |
|---|---|---|
| Fallback mean environment memakai test | Tidak; mean dibatasi `<= TRAIN_END` | Rendah |
| Lag/rolling melihat masa depan | Tidak terlihat; lag causal dan rolling memakai shift(1) | Rendah |
| Scaler fit pada validation/test | Tidak terlihat; fit memakai `[:n_tr]` | Rendah |
| Target `y_next` masuk fitur | Tidak terlihat | Rendah |
| Restart rollout validation memasukkan aktual validation setelah gap | Ya, secara eksplisit terjadi | **Tinggi** |
| Fitur lingkungan test mungkin tidak tersedia pada deployment | Belum ada availability contract | **Sedang–tinggi** |
| `rainfall_max_24h_mm` mungkin memakai window forward | Definisi sumber belum diaudit | **Sedang** |
| Reuse validation untuk model selection/eksperimen | Ya; dapat membuat estimasi final optimistis | Sedang |
| Graph/static data dibuat dari target/test | Graph dari koordinat/HydroRIVERS, bukan target | Rendah |
| Imputasi wl inference menggunakan masa depan | Tidak memakai bfill, tetapi kebijakan gap perlu diaudit | Rendah untuk leakage; tinggi untuk skew |

Sebelum refactor implementatif, provenance setiap kolom environment harus dikonfirmasi: waktu publikasi, apakah nilai aktual atau forecast, dan definisi window agregasinya.

## 7. Arsitektur Python tujuan

Struktur yang diusulkan (belum dibuat):

```text
notebook-graph-modular/
├── pyproject.toml
├── src/hydro_gnn/
│   ├── config.py
│   ├── data/
│   │   ├── io.py
│   │   ├── schema.py
│   │   └── artifacts.py
│   ├── graph/
│   │   ├── construction.py
│   │   └── validation.py
│   ├── features/
│   │   ├── environment.py
│   │   ├── water_level.py
│   │   ├── temporal.py
│   │   └── assembly.py
│   ├── datasets/
│   │   ├── windows.py
│   │   └── torch_dataset.py
│   ├── models/
│   │   ├── layers.py
│   │   ├── temporal_gnn.py
│   │   └── residual_temporal_gnn.py
│   ├── training/
│   │   ├── scaling.py
│   │   ├── engine.py
│   │   ├── validation.py
│   │   └── checkpoints.py
│   ├── inference/
│   │   ├── rollout.py
│   │   └── submission.py
│   └── visualization/
│       └── diagnostics.py
└── scripts/
    ├── build_dataset.py
    ├── train_baseline.py
    ├── train_residual_experiment.py
    ├── predict.py
    └── final_check.py
```

Prinsip batas modul:

- Fungsi transformasi data harus pure sejauh mungkin: input eksplisit, return eksplisit, tanpa membaca global notebook.
- Efek samping file hanya di `data/io.py`, `data/artifacts.py`, `training/checkpoints.py`, dan scripts.
- `config.py` menjadi satu sumber kebenaran; metadata config dan feature schema ikut disimpan dalam artifact/checkpoint.
- Rollout validation dan rollout inference menggunakan satu causal engine yang sama, dengan opsi warm-start dinyatakan eksplisit.
- Baseline dan residual memiliki kontrak model/checkpoint yang sama.
- Refactor awal harus mempertahankan output baseline sebelum perbaikan behavior dilakukan; bug fix dilakukan sebagai tahap terpisah dan diukur.

## 8. Mapping lengkap Cell → fungsi/unit → file Python tujuan

Nama fungsi di bawah adalah rancangan tujuan, bukan fungsi yang sudah dibuat.

| Cell | Peran sekarang | Fungsi/unit tujuan | File Python tujuan |
|---:|---|---|---|
| 0 | Heading import HydroRIVERS | Module docstring input spasial | `src/hydro_gnn/data/io.py` |
| 1 | Memilih root path Kaggle/lokal | `PathConfig`, `resolve_paths()` | `src/hydro_gnn/config.py` |
| 2 | Load shapefile HydroRIVERS | `load_hydrorivers()` | `src/hydro_gnn/data/io.py` |
| 3 | Sample print HydroRIVERS | `summarize_hydrorivers()` (diagnostic CLI) | `src/hydro_gnn/visualization/diagnostics.py` |
| 4 | Load koordinat dan head | `load_station_coordinates()` | `src/hydro_gnn/data/io.py` |
| 5 | Heading import train | Module docstring input target | `src/hydro_gnn/data/io.py` |
| 6 | Load train dan sample print | `load_train_data()` | `src/hydro_gnn/data/io.py` |
| 7 | Load data lingkungan | `load_environment_data()` | `src/hydro_gnn/data/io.py` |
| 8 | Sample print environment | `summarize_environment()` | `src/hydro_gnn/visualization/diagnostics.py` |
| 9 | Dokumentasi dataset/schema | `DatasetSchema` dan validator berbasis schema aktual | `src/hydro_gnn/data/schema.py` |
| 10 | Heading merge | Module docstring assembly | `src/hydro_gnn/features/assembly.py` |
| 11 | Heading parameter/periode | Module docstring konfigurasi | `src/hydro_gnn/config.py` |
| 12 | Periode, lag, rolling, graph config | `DataConfig`, `FeatureConfig`, `GraphConfig` | `src/hydro_gnn/config.py` |
| 13 | Heading urutan node | Module docstring schema node | `src/hydro_gnn/data/schema.py` |
| 14 | Canonical node order dan coverage check | `build_node_index()`, `validate_station_coverage()` | `src/hydro_gnn/data/schema.py` |
| 15 | Penjelasan konstruksi graph | Module docstring graph | `src/hydro_gnn/graph/construction.py` |
| 16 | Snap, traversal, fallback, edges, river attrs | `snap_stations_to_rivers()`, `trace_downstream_station()`, `build_river_graph()`, `extract_river_features()` | `src/hydro_gnn/graph/construction.py` |
| 17 | Penjelasan filter/imputasi environment | Module docstring preprocessing causal | `src/hydro_gnn/features/environment.py` |
| 18 | Observation grid dan imputasi environment | `build_observation_index()`, `prepare_environment_features()` | `src/hydro_gnn/features/environment.py` |
| 19 | Penjelasan lag/rolling | Module docstring fitur tinggi air | `src/hydro_gnn/features/water_level.py` |
| 20 | Lag dan rolling tinggi air | `build_water_level_features()` | `src/hydro_gnn/features/water_level.py` |
| 21 | Heading waktu/spasial | Module docstring fitur temporal | `src/hydro_gnn/features/temporal.py` |
| 22 | Fitur siklik waktu dan koordinat | `build_time_features()`, `build_spatial_features()` | `src/hydro_gnn/features/temporal.py` |
| 23 | Penjelasan assembly fitur/target | Module docstring assembly | `src/hydro_gnn/features/assembly.py` |
| 24 | Merge semua fitur dan `y_next` | `assemble_node_features()`, `build_next_step_target()`, `build_feature_schema()` | `src/hydro_gnn/features/assembly.py` |
| 25 | Penjelasan tensor/window | Module docstring dataset window | `src/hydro_gnn/datasets/windows.py` |
| 26 | Completeness, panel, target, train samples | `build_feature_panel()`, `find_valid_timesteps()`, `build_supervised_windows()` | `src/hydro_gnn/datasets/windows.py` |
| 27 | Penjelasan split/periode | Module docstring window policy | `src/hydro_gnn/datasets/windows.py` |
| 28 | Simpan NPZ | `DatasetArtifact`, `save_dataset_artifact()` | `src/hydro_gnn/data/artifacts.py`; orkestrasi di `scripts/build_dataset.py` |
| 29 | Heading debugging | Diagnostic command group | `src/hydro_gnn/visualization/diagnostics.py` |
| 30 | Missingness per node pada test | `missingness_by_station()` | `src/hydro_gnn/visualization/diagnostics.py` |
| 31 | Cek target train pada periode test | `validate_target_boundary()` | `src/hydro_gnn/data/schema.py` |
| 32 | Heading training | Script/module docstring | `scripts/train_baseline.py` |
| 33 | Seluruh baseline train/eval | Dipecah menjadi `load_dataset_artifact()`, `fit_scalers()`, `GraphTimeSeriesDataset`, `build_normalized_adjacency()`, `GCNLayer`, `TemporalGNN`, `train_epoch()`, `evaluate_epoch()`, `autoregressive_validate()`, `save_checkpoint()` | `data/artifacts.py`, `training/scaling.py`, `datasets/torch_dataset.py`, `graph/validation.py`, `models/layers.py`, `models/temporal_gnn.py`, `training/engine.py`, `training/validation.py`, `training/checkpoints.py`; orkestrasi `scripts/train_baseline.py` |
| 34 | Eksperimen residual dan plot comparison | `ResidualTemporalGNN`, shared trainer, `plot_training_history()`, `plot_model_comparison()` | `models/residual_temporal_gnn.py`, `training/engine.py`, `visualization/diagnostics.py`; orkestrasi `scripts/train_residual_experiment.py` |
| 35 | Heading inference | Script/module docstring | `scripts/predict.py` |
| 36 | Load model/data, causal rollout, submission | `load_checkpoint()`, `rebuild_water_window()`, `predict_autoregressive()`, `parse_submission_ids()`, `build_submission()`, `validate_submission()` | `training/checkpoints.py`, `inference/rollout.py`, `inference/submission.py`; orkestrasi `scripts/predict.py` |
| 37 | Plot cepat tiga node | `plot_station_forecasts()` | `src/hydro_gnn/visualization/diagnostics.py` |
| 38 | Heading final check | Script docstring | `scripts/final_check.py` |
| 39 | Load submission dan plot seluruh pos | `load_prediction_check_data()`, `plot_actual_vs_forecast_grid()` | `src/hydro_gnn/visualization/diagnostics.py`; orkestrasi `scripts/final_check.py` |

### 8.1 Mapping fungsi lintas-cell yang harus disatukan

Beberapa implementasi saat ini muncul lebih dari sekali dan harus memiliki satu source of truth:

| Implementasi notebook | Duplikasi | Tujuan tunggal |
|---|---|---|
| `GCNLayer`, `TemporalGNN` | Cell 33 dan 36 | `models/layers.py`, `models/temporal_gnn.py` |
| Normalisasi fitur inference/train | Cell 33 dan 36 | `training/scaling.py` dengan objek scaler serializable |
| Rebuild lag/rolling autoregresif | Cell 33 dan 36 | `inference/rollout.py` atau modul causal feature state bersama |
| Path/config periode/window | Cell 1, 12, 28, 33, 36 | `config.py` |
| Parsing/urutan schema fitur | Cell 24, 33, 36 | metadata artifact + `data/schema.py` |
| Checkpoint model | Cell 33 dan 34 | `training/checkpoints.py` dengan schema yang sama |

## 9. Tahapan refactor yang aman

### Tahap 0 — characterization sebelum ekstraksi

- Bekukan baseline dengan seed dan catat fingerprint input/config.
- Tambahkan test karakterisasi di branch refactor kelak, bukan di notebook audit ini.
- Rekam bentuk tensor, daftar fitur, node order, edge arrays, split datetime, scaler, prediksi beberapa step, dan submission hash/toleransi numerik.
- Bedakan test **behavior-preserving** dari perbaikan bug.

### Tahap 1 — ekstraksi tanpa perubahan perilaku

- Ekstrak config, loader, feature builders, graph builder, dataset artifact, model baseline, trainer, dan inference dengan urutan operasi yang sama.
- Pertahankan dtype, sorting, nilai default, rumus, dan checkpoint baseline.
- Jadikan scripts sebagai pengganti urutan Run All.

### Tahap 2 — kontrak dan validasi

- Tambahkan schema/frequency/uniqueness/finite checks.
- Simpan feature schema version, node order, config, dan fingerprint graph/dataset pada artifact/checkpoint.
- Tambahkan graph connectivity/collision diagnostics dan submission validation.

### Tahap 3 — perbaikan correctness terpisah

- Perbaiki/reset semantics rollout validation dan laporkan metric lama versus metric causal penuh.
- Satukan preprocessing missing water level training/inference.
- Hilangkan duplicated hard-coded config.
- Tangani identifier/kategori/cyclic features secara semantik setelah ablation.
- Setiap perubahan correctness harus menghasilkan perbandingan metric dan tidak disamarkan sebagai refactor mekanis.

### Tahap 4 — eksperimen dan observability

- Pisahkan residual experiment dari baseline production path.
- Tambahkan reproducibility, structured metrics, plots, dan manifest artifact.
- Pilih model inference berdasarkan config/registry eksplisit.

## 10. Kriteria selesai refactor mendatang

Refactor dapat dianggap selesai bila:

- Pipeline dapat dijalankan dari kernel/proses bersih melalui scripts tanpa notebook global.
- Semua cell memiliki rumah modul sesuai mapping.
- Dataset, graph, scaler, checkpoint, dan submission mempunyai schema/version/fingerprint yang tervalidasi.
- Baseline behavior-preserving lolos toleransi yang disepakati sebelum bug fix.
- Rollout validation dan inference memakai engine causal yang sama serta tidak menyisipkan aktual validation secara implisit.
- Tidak ada konfigurasi lag/window/periode/arsitektur yang diduplikasi.
- Training reproducible dengan seed dan menyimpan config lengkap.
- Submission memverifikasi ID, order, uniqueness, finite values, dan jumlah baris.
- Eksperimen residual tidak menimpa state baseline dan tidak wajib dua GPU secara hard-coded.
- Notebook, jika tetap dipertahankan, hanya menjadi thin orchestration/reporting layer.
