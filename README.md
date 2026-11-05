# Klasifikasi Jenis Sampah Berbasis Citra Menggunakan Convolutional Neural Network dengan Data Augmentation dan Teknik Regularisasi

**Mata kuliah:** Pembelajaran Mendalam · **Semester:** Ganjil TA 2026–2027 · **Kelompok:** Ethereal · **Kelas:** IFB-Sore
**Jenis pengumpulan:** Tugas Proyek 1

## Anggota & pembagian tugas

| No | NIM | Nama | Peran | Rincian pekerjaan | File |
|---|---|---|---|---|---|
| 1 | 231110923 | Sherly Yap | Eksplorasi data | Mengunduh dataset, distribusi kelas, komposisi sumber, contoh gambar, temuan EDA | `EDA.py` |
| 2 | 231110341 | Vincent C.Tio | Preprocessing & input pipeline | Cek gambar rusak & duplikat, resize 224x224, normalisasi, split 80/10/10, tf.data | `Preprocessing.py` |
| 3 | 231111124 | Felix Salim | Membangun & melatih CNN | Arsitektur CNN 4 blok, BatchNorm, Dropout, L2, class weight, callback | `CNN_Sampah.py` |
| 4 | 231110695 | Joel Adrianus Lase | Evaluasi & eksperimen perbaikan | Evaluasi baseline, 7 eksperimen perbaikan, studi ablasi, Grad-CAM | `evaluasi_model.py`, `eksperimen_perbaikan.py` |
| 5 | 231112730 | Jesslyn Sie | Laporan, dokumentasi, video | Menyusun laporan, README, video|


## Isi repository
| File / folder | Isi |
|---|---|
| `EDA.py`, `hasil_eda/` | Exploratory data analysis (Anggota 1) |
| `Preprocessing.py` | Preprocessing & input pipeline (Anggota 2) |
| `CNN_Sampah.py` | Arsitektur & training CNN baseline (Anggota 3) |
| `evaluasi_model.py`, `eksperimen_perbaikan.py` | Evaluasi & eksperimen perbaikan (Anggota 4) |
| `notebooks/` | Notebook Google Colab (cara menjalankan + output) |
| `cnn_training_output/` | Hasil training baseline |
| `hasil_eksperimen/` | Hasil evaluasi & semua eksperimen |
| `datasett_preprocessed/` | Dataset (tidak disertakan, lihat `BACA_SAYA.txt`) |

## Dataset
Gabungan dua dataset citra sampah dunia nyata, 5.061 gambar, 10 kelas, 224x224 piksel (PNG):
- **TrashNet** (cardboard, glass, metal, paper, trash): https://github.com/garythung/trashnet
- **RealWaste** (Food Organics, Miscellaneous Trash, Plastic, Textile Trash, Vegetation, serta sebagian cardboard/glass/metal/paper): https://archive.ics.uci.edu/dataset/908/realwaste


## Cara menjalankan
Semua perintah dijalankan dari folder utama repository ini. Instal dependensi: `pip install -r requirements.txt`.
Bagian Anggota 4 dijalankan di **Google Colab (GPU T4)** karena laptop tidak mendukung GPU; notebook di folder
`notebooks/` berisi langkah menjalankan beserta output-nya.

0. EDA: `python EDA.py`
1. Preprocessing: `python Preprocessing.py`
2. Training CNN baseline: `python CNN_Sampah.py`
3. Evaluasi baseline: `python evaluasi_model.py`
4. Eksperimen perbaikan (Colab: `notebooks/Colab_Evaluasi_Eksperimen.ipynb`)
   ```
   python eksperimen_perbaikan.py --exp E1 E2 A1 A2 A3 A4 A5
   python eksperimen_perbaikan.py --ringkasan
   ```

## Hasil utama (test set, 516 gambar)

| Eksperimen | Keterangan | Akurasi test | Macro F1 | Akurasi val |
|---|---|---|---|---|
| E0 | CNN 4 blok dari nol (baseline Anggota 3) | 0,554 | 0,541 | 0,573 |
| E1 | CNN yang sama + AdamW, warmup-cosine LR, BN momentum 0.9, label smoothing, augmentasi kuat | 0,661 | 0,639 | 0,697 |
| E2 | Transfer learning MobileNetV2 + augmentasi + regularisasi + fine-tuning | 0,828 | 0,811 | 0,830 |
| A1 | MobileNetV2 beku, tanpa augmentasi & regularisasi | 0,839 | 0,824 | 0,860 |
| A2 | MobileNetV2 beku, augmentasi saja | 0,806 | 0,778 | 0,820 |
| **A3** | **MobileNetV2 beku, regularisasi saja (MODEL FINAL)** | **0,833** | **0,817** | **0,862** |
| A4 | MobileNetV2 beku, augmentasi + regularisasi | 0,816 | 0,792 | 0,830 |
| A5 | Sama dengan A3 tetapi **tanpa class weight** (pembanding fitur class weight) | 0,826 | 0,803 | 0,872 |

Model final (A3) dipilih dari kandidat E1, E2, dan A1–A4 berdasarkan akurasi **validation** tertinggi,
bukan akurasi test, agar data test tetap objektif. A5 dilatih **setelah** pemilihan, khusus untuk mengukur
dampak class weight. Walaupun akurasi validation A5 sedikit lebih tinggi, A5 tidak dipakai karena tanpa
class weight recall kelas terkecil (`trash`) turun dari 1,00 menjadi 0,73; pada data tidak seimbang,
akurasi keseluruhan lebih dipengaruhi kelas besar.


## Fitur unik
| No | Fitur | Bukti dampak |
|---|---|---|
| F1 | Grad-CAM | Peta perhatian model pada gambar test (`hasil_eksperimen/*/gradcam.png`) |
| F2 | Penanganan data tidak seimbang (class weight) | A3 (dengan class weight) vs A5 (tanpa): Macro F1 0,817 vs 0,803; recall kelas terkecil `trash` 1,00 vs 0,73 (15 gambar test) |


## Tautan
| Source code (GitHub) | https://github.com/xiitkj76/klasifikasi-sampah-ifb-ethereal.git |

