# M1 – Image Captioning dengan CLIP ViT-B/32

Notebook `pipeline-image-captioning-clip-vit-b-32.ipynb` digunakan untuk menghitung embedding gambar dan teks menggunakan CLIP, mengevaluasi kesamaan gambar–caption, serta melakukan pencarian gambar berdasarkan teks (_text-to-image retrieval_).

## Struktur proyek

Pastikan struktur folder proyek mengikuti susunan berikut:

```text
TugasAkhir/
├── data/
│   ├── images/                 # File gambar yang dirujuk oleh CSV
│   ├── train_clean.csv
│   └── val_clean.csv
├── M1/
│   ├── pipeline-image-captioning-clip-vit-b-32.ipynb
│   └── README.md               # File ini
└── Output/                     # Lokasi hasil evaluasi
```

`val_clean.csv` perlu memiliki kolom `file`, `caption_id`, dan `caption_en`. Nilai pada kolom `file` harus sesuai dengan nama file gambar di folder `images/`. Notebook saat ini menggunakan data validasi (`val_clean.csv`); `train_clean.csv` dan `PATH_TRAIN` disiapkan sebagai referensi, tetapi belum digunakan dalam proses notebook.

## 1. Menjalankan di Kaggle Notebook (utama)

1. Buat notebook baru di Kaggle, lalu tambahkan tiga dataset melalui bagian **Input**: dataset gambar, data train, dan data validasi.
2. Pastikan isi dataset memiliki file `train_clean.csv`, `val_clean.csv`, dan folder yang berisi gambar.
3. Buka notebook ini di Kaggle atau unggah file `.ipynb` ke notebook Kaggle.
4. Pastikan **Internet** di pengaturan notebook aktif agar model CLIP dapat diunduh dari Hugging Face saat pertama kali dijalankan. GPU dapat diaktifkan untuk membantu mempercepat proses.
5. Periksa dan sesuaikan konfigurasi path pada bagian konfigurasi dataset jika lokasi input Kaggle berbeda:

```python
PATH_TRAIN = "/kaggle/input/datasets/geogeogeogeogeo/train-data/"
PATH_VAL = "/kaggle/input/datasets/geogeogeogeogeo/validation-data/"
PATH_IMAGES = "/kaggle/input/datasets/geogeogeogeogeo/images/"
PATH_OUTPUT = "/kaggle/working/hasil_evaluasi.json"
Note: sesuaikan dengan path anda. "/kaggle/input/datasets/geogeogeogeogeo" adalah path penulis.
```

`PATH_IMAGES` harus menunjuk langsung ke folder tempat file gambar berada. Jika file gambar tersimpan di subfolder `images/`, tambahkan subfolder tersebut pada path. Untuk memeriksa lokasi file yang tersedia, jalankan:

```python
!find /kaggle/input -maxdepth 5 -type f | head -30
```

6. Jalankan seluruh sel notebook secara berurutan (**Run All**). Hasil evaluasi akan disimpan di `/kaggle/working/hasil_evaluasi.json`. Unduh file tersebut dari bagian **Output** apabila ingin menyimpannya ke komputer.

## 2. Menjalankan di Google Colab

1. Unggah folder proyek `TugasAkhir` ke Google Drive dengan struktur folder seperti di atas.
2. Buka file notebook melalui Google Colab.
3. Pasang Google Drive dan sesuaikan konfigurasi path. Contoh jika folder proyek berada di `MyDrive/TugasAkhir`:

```python
from google.colab import drive
import os

drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/TugasAkhir"
PATH_TRAIN = f"{BASE_DIR}/data/"
PATH_VAL = f"{BASE_DIR}/data/"
PATH_IMAGES = f"{BASE_DIR}/data/images/"
PATH_OUTPUT = f"{BASE_DIR}/Output/hasil_evaluasi.json"
os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)
```

4. Jika pustaka `sentence-transformers` belum tersedia, jalankan:

```python
!pip install -q sentence-transformers
```

5. Jalankan seluruh sel notebook secara berurutan. File hasil akan tersimpan di folder `Output` pada Google Drive.

## 3. Menjalankan secara lokal (IDE seperti VS Code/Jupyter)

1. Simpan atau clone folder proyek `TugasAkhir` ke komputer.
2. Buka notebook dari folder `M1`. Konfigurasi path berikut mengasumsikan _working directory_ berada di folder `M1`:

```python
PATH_TRAIN = "../data/"
PATH_VAL = "../data/"
PATH_IMAGES = "../data/images/"
PATH_OUTPUT = "../Output/hasil_evaluasi.json"
```

3. Siapkan environment Python, kemudian instal pustaka yang diperlukan:

```bash
pip install numpy torch transformers sentence-transformers pillow pandas matplotlib ipython
```

Jika PyTorch belum terinstal atau ingin menggunakan GPU, instal versi PyTorch yang sesuai dengan sistem operasi dan CUDA melalui panduan resmi PyTorch. 4. Jalankan seluruh sel notebook secara berurutan. Pastikan komputer terhubung ke internet saat model CLIP pertama kali diunduh. Hasil evaluasi akan disimpan ke folder `Output`.

> **Catatan:** Path relatif lokal bergantung pada _working directory_ notebook. Jika notebook dijalankan dari folder selain `M1`, sesuaikan kembali `PATH_TRAIN`, `PATH_VAL`, `PATH_IMAGES`, dan `PATH_OUTPUT`.

## Hasil yang dihasilkan

Notebook menampilkan matriks kesamaan kosinus, nilai **Recall@1/5/10** dan **mAP@1/5/10** untuk pencarian gambar–caption dalam bahasa Indonesia dan Inggris, serta visualisasi hasil _text-to-image retrieval_ berupa tiga gambar teratas untuk setiap kueri inferensi. Ringkasan waktu eksekusi dan metrik evaluasi disimpan dalam `hasil_evaluasi.json`.
