# Ringkasan EDA

- Total gambar: **5061**, jumlah kelas: **10**
- Ukuran gambar: 224x224 piksel; mode warna: RGB
- Gambar rusak: 0; file duplikat identik: 0
- Gambar dari TrashNet: 2045 (40.4%); dari RealWaste: 3016 (59.6%)

## Distribusi kelas
| Kelas | Jumlah | % | TrashNet | RealWaste |
|---|---|---|---|---|
| cardboard | 498 | 9.8% | 404 | 94 |
| Food Organics | 411 | 8.1% | 0 | 411 |
| glass | 586 | 11.6% | 501 | 85 |
| metal | 567 | 11.2% | 409 | 158 |
| Miscellaneous Trash | 495 | 9.8% | 0 | 495 |
| paper | 694 | 13.7% | 594 | 100 |
| Plastic | 921 | 18.2% | 0 | 921 |
| Textile Trash | 318 | 6.3% | 0 | 318 |
| trash | 137 | 2.7% | 137 | 0 |
| Vegetation | 434 | 8.6% | 0 | 434 |

## Temuan penting yang memengaruhi preprocessing & pemodelan
1. **Data tidak seimbang.** Kelas terbesar `Plastic` (921 gambar) sekitar 6.7 kali kelas terkecil `trash` (137 gambar). Karena itu pembagian data dibuat per kelas (stratified 80/10/10), training memakai class weight, dan evaluasi memakai Macro F1 selain akurasi.
2. **Dua sumber dengan latar foto berbeda.** Kelas `Food Organics`, `Miscellaneous Trash`, `Plastic`, `Textile Trash`, `Vegetation` 100% dari RealWaste; kelas `trash` 100% dari TrashNet; kelas `cardboard`, `glass`, `metal`, `paper` campuran (mayoritas TrashNet). Kecerahannya hampir sama, tetapi **warna latarnya berbeda**: latar TrashNet cenderung krem/kecokelatan (rata-rata R-B di tepi gambar +0.073), sedangkan RealWaste abu-abu kebiruan (-0.041); terlihat jelas pada rata2_gambar_per_kelas.png. Model berisiko mengenali warna latar/sumber foto, bukan jenis sampahnya (dicek kemudian dengan Grad-CAM dan akurasi per sumber).
3. **Kemiripan makna antarkelas.** `trash` (TrashNet) dan `Miscellaneous Trash` (RealWaste) sama-sama sampah campuran, sehingga berpotensi sering tertukar. Dari pengamatan contoh_per_kelas.png, kelas `trash` juga banyak berisi kemasan/plastik bekas sehingga mirip kelas `Plastic`.
4. **Variasi kondisi citra.** Ukuran gambar sudah seragam 224x224 hasil preprocessing, tetapi posisi objek, sudut, dan pencahayaan bervariasi, sehingga augmentasi geometri dan warna relevan untuk dicoba.