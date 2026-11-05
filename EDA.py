"""
Penanggung jawab : Anggota 1 - Sherly Yap 231110923

EXPLORATORY DATA ANALYSIS (EDA) DATASET SAMPAH

Menganalisis folder datasett_preprocessed/ (10 folder kelas) dan menghasilkan bahan
untuk laporan bab 3 (EDA):
  a. distribusi kelas                       -> distribusi_kelas.png
  e. contoh sampel per kelas                -> contoh_per_kelas.png
  +  komposisi sumber dataset per kelas      -> komposisi_sumber.png
  +  rata-rata gambar per kelas (pola latar) -> rata2_gambar_per_kelas.png
  +  warna latar (tepi gambar) per sumber    -> warna_latar_per_sumber.png
  f. temuan penting                          -> ringkasan_eda.md
  +  tabel statistik                         -> statistik_kelas.csv

Hanya membutuhkan numpy, pillow, dan matplotlib (tanpa TensorFlow).

Cara pakai (dari folder yang sama dengan CNN_Sampah.py):
  python EDA.py
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasett_preprocessed"
OUT = BASE_DIR / "hasil_eda"
EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
WARNA = {"TrashNet": "#4C72B0", "RealWaste": "#DD8452"}


def sumber_gambar(path: Path) -> str:
    """RealWaste memakai nama 'Kelas_123', TrashNet memakai 'kelas123'."""
    return "RealWaste" if re.search(r"_\d+$", path.stem) else "TrashNet"


def kumpulkan():
    kelas = sorted((p for p in DATASET_DIR.iterdir() if p.is_dir()), key=lambda p: (p.name.casefold(), p.name))
    data = {}
    for k in kelas:
        data[k.name] = sorted((p for p in k.rglob("*") if p.suffix.lower() in EKSTENSI),
                              key=lambda p: p.name.casefold())
    return data


def main():
    if not DATASET_DIR.is_dir():
        raise SystemExit(f"Folder dataset tidak ditemukan: {DATASET_DIR}")
    OUT.mkdir(exist_ok=True)
    data = kumpulkan()
    nama = list(data)
    jumlah = np.array([len(v) for v in data.values()])
    total = int(jumlah.sum())

    # ---------------------------------------------------------------- statistik per gambar
    rng = random.Random(SEED)
    statistik, ukuran, mode, rusak, hash_seen, duplikat = {}, set(), set(), [], {}, []
    warna_tepi = {"TrashNet": [], "RealWaste": []}   # rata-rata (R - B) di pita tepi gambar
    rata2 = {}
    for k, paths in data.items():
        sumber = [sumber_gambar(p) for p in paths]
        akum, terang, rgb = np.zeros((224, 224, 3)), [], []
        for p in paths:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h in hash_seen:
                duplikat.append((str(p), hash_seen[h]))
            hash_seen.setdefault(h, str(p))
            try:
                with Image.open(p) as img:
                    ukuran.add(img.size)
                    mode.add(img.mode)
                    a = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
            except OSError as e:
                rusak.append((str(p), str(e)))
                continue
            akum += a
            terang.append(a.mean())
            rgb.append(a.reshape(-1, 3).mean(0))
            tepi = np.concatenate([a[:20].reshape(-1, 3), a[-20:].reshape(-1, 3),
                                   a[:, :20].reshape(-1, 3), a[:, -20:].reshape(-1, 3)])
            warna_tepi[sumber_gambar(p)].append(float(tepi[:, 0].mean() - tepi[:, 2].mean()))
        rata2[k] = akum / max(len(terang), 1)
        rgb = np.array(rgb)
        statistik[k] = {
            "jumlah": len(paths),
            "persen": len(paths) / total * 100,
            "trashnet": sumber.count("TrashNet"),
            "realwaste": sumber.count("RealWaste"),
            "kecerahan_rata2": float(np.mean(terang)),
            "r": float(rgb[:, 0].mean()), "g": float(rgb[:, 1].mean()), "b": float(rgb[:, 2].mean()),
        }

    # ---------------------------------------------------------------- tabel
    with (OUT / "statistik_kelas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kelas", "jumlah", "persen", "dari_TrashNet", "dari_RealWaste",
                    "kecerahan_rata2", "rata2_R", "rata2_G", "rata2_B"])
        for k, s in statistik.items():
            w.writerow([k, s["jumlah"], f"{s['persen']:.1f}", s["trashnet"], s["realwaste"],
                        f"{s['kecerahan_rata2']:.3f}", f"{s['r']:.3f}", f"{s['g']:.3f}", f"{s['b']:.3f}"])

    # ---------------------------------------------------------------- a. distribusi kelas
    urut = np.argsort(-jumlah)
    fig, ax = plt.subplots(figsize=(12, 5))
    b = ax.bar([nama[i] for i in urut], jumlah[urut], color="#55A868")
    ax.bar_label(b, labels=[f"{jumlah[i]}\n({jumlah[i] / total:.1%})" for i in urut], fontsize=8)
    ax.axhline(jumlah.mean(), color="gray", ls="--", label=f"Rata-rata = {jumlah.mean():.0f}")
    ax.set_ylabel("Jumlah gambar")
    ax.set_title(f"Distribusi kelas ({total} gambar, rasio terbesar:terkecil = "
                 f"{jumlah.max() / jumlah.min():.1f} : 1)")
    ax.set_ylim(0, jumlah.max() * 1.18)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "distribusi_kelas.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- komposisi sumber
    tn = np.array([statistik[k]["trashnet"] for k in nama])
    rw = np.array([statistik[k]["realwaste"] for k in nama])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(nama, tn, color=WARNA["TrashNet"], label="TrashNet")
    ax.bar(nama, rw, bottom=tn, color=WARNA["RealWaste"], label="RealWaste")
    for i in range(len(nama)):
        ax.text(i, tn[i] + rw[i] + 8, f"{tn[i] / (tn[i] + rw[i]):.0%} TN", ha="center", fontsize=8)
    ax.set_ylabel("Jumlah gambar")
    ax.set_title("Komposisi sumber dataset per kelas")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "komposisi_sumber.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- e. contoh per kelas
    n = 5
    fig, axes = plt.subplots(len(nama), n, figsize=(n * 2.1, len(nama) * 2.2))
    for r, k in enumerate(nama):
        pilih = rng.sample(data[k], n)
        for c, p in enumerate(pilih):
            ax = axes[r, c]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(WARNA[sumber_gambar(p)])
                s.set_linewidth(2.5)
            if c == 0:
                ax.set_ylabel(k, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle("Contoh gambar per kelas (bingkai biru = TrashNet, oranye = RealWaste)", fontsize=11)
    fig.tight_layout(rect=(0.04, 0, 1, 0.98))
    fig.savefig(OUT / "contoh_per_kelas.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- rata-rata gambar
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
    for ax, k in zip(axes.ravel(), nama):
        ax.imshow(np.clip(rata2[k], 0, 1))
        ax.set_title(f"{k}\n(TrashNet {statistik[k]['trashnet'] / statistik[k]['jumlah']:.0%})", fontsize=9)
        ax.axis("off")
    fig.suptitle("Rata-rata semua gambar per kelas: menunjukkan pola latar dan posisi objek", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "rata2_gambar_per_kelas.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- warna latar per sumber
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s_, v in warna_tepi.items():
        ax.hist(v, bins=40, alpha=0.65, density=True, color=WARNA[s_],
                label=f"{s_} (n={len(v)}, rata-rata {np.mean(v):+.3f})")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Warna latar di tepi gambar: R - B  (negatif = kebiruan, positif = kecokelatan/hangat)")
    ax.set_ylabel("Kepadatan")
    ax.set_title("Warna latar foto berbeda antara TrashNet dan RealWaste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "warna_latar_per_sumber.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- ringkasan
    besar, kecil = nama[int(jumlah.argmax())], nama[int(jumlah.argmin())]
    hanya_tn = [k for k in nama if statistik[k]["realwaste"] == 0]
    hanya_rw = [k for k in nama if statistik[k]["trashnet"] == 0]
    campur = [k for k in nama if k not in hanya_tn + hanya_rw]
    md = [
        "# Ringkasan EDA", "",
        f"- Total gambar: **{total}**, jumlah kelas: **{len(nama)}**",
        f"- Ukuran gambar: {', '.join(f'{w}x{h}' for w, h in sorted(ukuran))} piksel; mode warna: {', '.join(sorted(mode))}",
        f"- Gambar rusak: {len(rusak)}; file duplikat identik: {len(duplikat)}",
        f"- Gambar dari TrashNet: {int(tn.sum())} ({tn.sum() / total:.1%}); dari RealWaste: {int(rw.sum())} ({rw.sum() / total:.1%})",
        "", "## Distribusi kelas", "| Kelas | Jumlah | % | TrashNet | RealWaste |", "|---|---|---|---|---|",
    ]
    md += [f"| {k} | {statistik[k]['jumlah']} | {statistik[k]['persen']:.1f}% | {statistik[k]['trashnet']} | "
           f"{statistik[k]['realwaste']} |" for k in nama]
    md += [
        "", "## Temuan penting yang memengaruhi preprocessing & pemodelan",
        f"1. **Data tidak seimbang.** Kelas terbesar `{besar}` ({jumlah.max()} gambar) sekitar "
        f"{jumlah.max() / jumlah.min():.1f} kali kelas terkecil `{kecil}` ({jumlah.min()} gambar). "
        "Karena itu pembagian data dibuat per kelas (stratified 80/10/10), training memakai class weight, "
        "dan evaluasi memakai Macro F1 selain akurasi.",
        f"2. **Dua sumber dengan latar foto berbeda.** Kelas {', '.join(f'`{k}`' for k in hanya_rw)} 100% dari RealWaste; "
        f"kelas {', '.join(f'`{k}`' for k in hanya_tn)} 100% dari TrashNet; kelas "
        f"{', '.join(f'`{k}`' for k in campur)} campuran (mayoritas TrashNet). "
        "Kecerahannya hampir sama, tetapi **warna latarnya berbeda**: latar TrashNet cenderung krem/kecokelatan "
        f"(rata-rata R-B di tepi gambar {np.mean(warna_tepi['TrashNet']):+.3f}), sedangkan RealWaste abu-abu kebiruan "
        f"({np.mean(warna_tepi['RealWaste']):+.3f}); terlihat jelas pada rata2_gambar_per_kelas.png. "
        "Model berisiko mengenali warna latar/sumber foto, bukan jenis sampahnya "
        "(dicek kemudian dengan Grad-CAM dan akurasi per sumber).",
        "3. **Kemiripan makna antarkelas.** `trash` (TrashNet) dan `Miscellaneous Trash` (RealWaste) sama-sama sampah "
        "campuran, sehingga berpotensi sering tertukar. Dari pengamatan contoh_per_kelas.png, kelas `trash` juga "
        "banyak berisi kemasan/plastik bekas sehingga mirip kelas `Plastic`.",
        f"4. **Variasi kondisi citra.** Ukuran gambar sudah seragam {', '.join(f'{w}x{h}' for w, h in sorted(ukuran))} "
        "hasil preprocessing, tetapi posisi objek, sudut, dan pencahayaan bervariasi, sehingga augmentasi geometri "
        "dan warna relevan untuk dicoba.",
    ]
    (OUT / "ringkasan_eda.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSemua hasil EDA tersimpan di: {OUT}")


if __name__ == "__main__":
    main()
