# Studi Kasus 6: Atribusi Pemicu Deforestasi Menggunakan Citra Satelit Multi-Sensor & AI

Proyek ini disusun berdasarkan materi kuliah praktisi **"Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI"** yang disampaikan oleh **Dr. Bart Slaughter** (Wageningen University and Research, Belanda).

## Deskripsi Proyek
Proyek ini mengimplementasikan pemodelan *Machine Learning* untuk mengklasifikasikan pemicu langsung (*direct drivers*) dari peringatan deforestasi (*deforestation alerts*) menggunakan data gabungan citra satelit radar (Sentinel-1) dan optik (Sentinel-2). Model membedakan pemicu seperti pertanian skala kecil/besar, pertambangan, pembalakan liar (*selective logging*), dan kebakaran hutan (*wildfires*).

## Pustaka & Teknologi yang Digunakan
- **Python**
- **Pandas** & **NumPy** (untuk ekstraksi dan simulasi fitur geospasial)
- **Scikit-Learn** (untuk pelatihan model klasifikasi Random Forest dan penilaian akurasi/F1-score)

## Cara Menjalankan Program
1. Install dependensi dengan menjalankan perintah berikut di terminal:
   ```bash
   pip install -r requirements.txt
