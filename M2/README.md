# M2 – Eksperimen Varian Model Vision-Language untuk Cross-Modal Retrieval

Notebook `experiment.ipynb` digunakan untuk mengevaluasi dan membandingkan performa model baseline (M1) dengan 3 varian model vision-language multilingual lainnya dalam tugas *cross-modal retrieval* (Image-to-Text dan Text-to-Image) pada bahasa Indonesia dan bahasa Inggris.

---

## Daftar Varian Model yang Dieksperimenkan

Eksperimen pada M2 mencakup 4 arsitektur model dengan karakteristik berbeda:

| Varian | Nama Model / Checkpoint | Backend | Dimensi Vektor | Catatan Khusus |
|---|---|---|---|---|
| **Baseline M1** | `sentence-transformers/clip-ViT-B-32` (Image)<br>`sentence-transformers/clip-ViT-B-32-multilingual-v1` (Text) | SentenceTransformers | 512 | Model baseline dari tahap M1 |
| **Varian 1** | `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k` | OpenCLIP | 512 | Vision ViT-B-32 dipasangkan dengan text encoder XLM-RoBERTa base (LAION-5B) |
| **Varian 2** | `google/siglip2-base-patch16-224` | Transformers (HF) | 768 | Model SigLIP 2 terbaru dari Google berbasis *sigmoid loss* |
| **Varian 3** | `visheratin/nllb-clip-base-siglip` | OpenCLIP | 768 | Menggunakan tokenizer NLLB-200 multilingual dengan kode bahasa (`ind_Latn` & `eng_Latn`) |

---

## Alur Pipeline Eksperimen

1. **Persiapan Dependensi & Dataset**: Memuat dataset validasi `val_clean.csv` (berisi kolom `file`, `caption_id`, `caption_en`) beserta file gambar terkait.
2. **Smoke Test (Sanity Check)**: Pengujian cepat pada 2 sampel data untuk seluruh varian model guna memastikan:
   - Model berhasil dimuat tanpa galat (*error-free*).
   - Tidak ada nilai `NaN` atau `Inf` pada vektor embedding.
   - Embedding telah ternormalisasi L2 ($\|v\|_2 = 1.0$).
3. **Ekstraksi Embedding Penuh**:
   - Mengekstrak embedding gambar ($E$), caption bahasa Indonesia ($T_i$), dan caption bahasa Inggris ($T_e$).
   - Dilengkapi sistem *caching* `.pt` dan manajemen memori RAM (`gc.collect()` serta penghapusan model yang selesai diproses) agar terhindar dari *Out-of-Memory* (OOM).
4. **Penyimpanan Artefak & Verifikasi Integritas**:
   - Menyimpan embedding per varian (`E.npy`, `Ti.npy`, `Te.npy`, `metadata.json`).
   - Menyusun ringkasan dimensi dan waktu ekstraksi ke `experiment_summary.csv`.
   - Mengompres seluruh artefak ke dalam `M2_Experiment_Results.zip`.
   - Melakukan uji asersi otomatis terhadap integritas file ZIP dan validitas dimensi array.
5. **Evaluasi Cross-Modal Retrieval**:
   - Menghitung matriks kesamaan kosinus (*Cosine Similarity*).
   - Menghitung metrik **Recall@1**, **Recall@5**, **Recall@10** serta **mAP@1**, **mAP@5**, **mAP@10**.
   - Mengevaluasi 4 arah retrieval:
     - Gambar → Teks Bahasa Indonesia
     - Teks Bahasa Indonesia → Gambar
     - Gambar → Teks Bahasa Inggris
     - Teks Bahasa Inggris → Gambar
   - Menyimpan metrik evaluasi ke `hasil_evaluasi.json`.
6. **Inference & Visualisasi Top-3**:
   - Melakukan pencarian gambar berdasarkan 10 kueri teks inferensi bernuansa lokal Indonesia.
   - Menampilkan visualisasi grid Top-3 gambar teratas dengan skor kesamaan kosinus dan verifikasi *ground truth*.

---

## Struktur Proyek

Pastikan struktur folder proyek mengikuti susunan berikut:

```text
Proyek-Akhir-PMM-Sigma/
├── data/
│   ├── images/                 # File-file gambar yang dirujuk oleh CSV
│   ├── train_clean.csv         # Data latih (referensi)
│   └── val_clean.csv           # Data validasi (file, caption_id, caption_en)
├── M1/
│   ├── pipeline-image-captioning-clip-vit-b-32.ipynb
│   └── README.md
├── M2/
│   ├── experiment.ipynb        # Notebook eksperimen utama M2
│   └── README.md               # File panduan ini
└── Output/                     # Lokasi keluaran hasil evaluasi & ekspor
```

---

## Petunjuk Menjalankan

### 1. Menjalankan di Kaggle Notebook (Direkomendasikan)

1. Buat notebook baru di Kaggle dan unggah file `experiment.ipynb`.
2. Tambahkan dataset gambar dan metadata CSV ke bagian **Input**.
3. Pastikan opsi **Internet** aktif (*ON*) di panel *Settings* agar bobot model dan tokenizer dapat diunduh.
4. Sesuaikan konfigurasi path pada sel konfigurasi (Cell 6):

```python
PATH_TRAIN  = "/kaggle/input/datasets/stinart/cleaned-sea-vl-indonesian-only/"
PATH_VAL    = "/kaggle/input/datasets/stinart/cleaned-sea-vl-indonesian-only/"
PATH_IMAGES = "/kaggle/input/datasets/stinart/cleaned-sea-vl-indonesian-only/"
PATH_OUTPUT = "/kaggle/working/hasil_evaluasi.json"
PATH_CACHE  = "/kaggle/working/cache_m2/"
EXPORT_DIR  = "/kaggle/working/M2_Experiment_Results"
```

5. Jalankan seluruh sel notebook secara berurutan (**Run All**).
6. Unduh artefak `M2_Experiment_Results.zip` dan `hasil_evaluasi.json` dari panel **Output** Kaggle.

---

### 2. Menjalankan di Google Colab

1. Unggah folder proyek ke Google Drive.
2. Buka file `experiment.ipynb` melalui Google Colab.
3. Sambungkan (*mount*) Google Drive dan atur konfigurasi path:

```python
from google.colab import drive
import os

drive.mount('/content/drive')

BASE_DIR    = "/content/drive/MyDrive/Proyek-Akhir-PMM-Sigma"
PATH_TRAIN  = f"{BASE_DIR}/data/"
PATH_VAL    = f"{BASE_DIR}/data/"
PATH_IMAGES = f"{BASE_DIR}/data/"
PATH_OUTPUT = f"{BASE_DIR}/Output/hasil_evaluasi.json"
PATH_CACHE  = f"{BASE_DIR}/Output/cache_m2/"
EXPORT_DIR  = f"{BASE_DIR}/Output/M2_Experiment_Results"

os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)
os.makedirs(PATH_CACHE, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
```

4. Instal dependensi tambahan yang dibutuhkan:

```bash
!pip install -q --upgrade open_clip_torch transformers sentence-transformers sentencepiece protobuf
```

5. Jalankan seluruh sel secara berurutan.

---

### 3. Menjalankan secara Lokal (VS Code / JupyterLab)

1. Buka folder repositori di IDE pilihan Anda.
2. Pastikan virtual environment Python (>= 3.9) telah aktif, lalu instal paket dependensi:

```bash
pip install torch torchvision transformers sentence-transformers open_clip_torch sentencepiece protobuf pandas numpy pillow matplotlib ipython
```

3. Pada notebook `experiment.ipynb`, gunakan konfigurasi path relatif lokal:

```python
PATH_TRAIN  = "../data/"
PATH_VAL    = "../data/"
PATH_IMAGES = "../data/"
PATH_OUTPUT = "../Output/hasil_evaluasi.json"
PATH_CACHE  = "../Output/cache_m2/"
EXPORT_DIR  = "../Output/M2_Experiment_Results"
```

4. Jalankan seluruh sel notebook secara berurutan.

---

## Hasil dan Artefak yang Dihasilkan

Setelah notebook selesai dijalankan, artefak berikut akan dibuat:

1. **`M2_Experiment_Results.zip`**: Arsip berisi data embedding dan metadata seluruh varian:
   - Subfolder tiap model (`baseline_m1/`, `varian1_laion5b/`, `varian2_siglip2/`, `varian3_nllb_clip/`):
     - `E.npy`: Vektor embedding gambar ($N \times D$, ternormalisasi L2).
     - `Ti.npy`: Vektor embedding teks bahasa Indonesia ($N \times D$, ternormalisasi L2).
     - `Te.npy`: Vektor embedding teks bahasa Inggris ($N \times D$, ternormalisasi L2).
     - `metadata.json`: Informasi checkpoint, dimensi vektor, konfigurasi tokenizer, dan alignment urutan ID sampel.
   - `experiment_summary.csv`: Tabel ringkasan perbandingan dimensi dan durasi waktu komputasi ekstraksi.
2. **`hasil_evaluasi.json`**: Berkas metrik komprehensif berisi nilai Recall@1/5/10 dan mAP@1/5/10 untuk ke-4 arah retrieval serta total waktu eksekusi.
3. **Visualisasi Inferensi Top-3**: Plot grid gambar yang menampilkan 3 kandidat teratas untuk setiap kueri teks inferensi, lengkap dengan skor kesamaan kosinus dan status kecocokan (*Benar / Salah*).
