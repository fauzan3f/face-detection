## Penjelasan Singkat

Aplikasi ini merupakan **Aplikasi Pengolahan Citra Digital** berbasis Python dengan antarmuka grafis (GUI) menggunakan PyQt5. Aplikasi ini memiliki fitur utama berupa ekstraksi ciri (feature extraction) dan deteksi objek (object detection) pada gambar. Pengguna dapat memuat gambar dari file atau kamera, melakukan analisis tekstur, bentuk, dan warna menggunakan berbagai metode (Histogram, GLCM, Haralick, Hu Moments, LBP, HOG), serta mendeteksi objek seperti wajah, mata, warna dominan, kontur, garis, dan sudut. Hasil analisis dapat disimpan dalam berbagai format untuk keperluan dokumentasi atau analisis lanjutan.

# Alur Kerja Sistem dan Penjelasan Fitur

Aplikasi ini terdiri dari beberapa modul utama yang saling terintegrasi untuk mendukung proses pengolahan citra digital:

### 1. Modul Input Gambar
Modul ini menangani proses akuisisi gambar ke dalam sistem melalui dua metode utama:
- **Pemuatan File**: Pengguna dapat memuat gambar dari penyimpanan komputer.
- **Pengambilan Kamera**: Pengguna dapat mengambil gambar langsung dari kamera yang terhubung.

### 2. Modul Penyimpanan Gambar
- **Penyimpanan Internal**: Menyimpan gambar asli, gambar hasil pemrosesan, dan gambar sementara.
- **Penyimpanan Eksternal**: Pengguna dapat menyimpan gambar hasil pemrosesan, data piksel, dan hasil ekstraksi ciri ke berbagai format file (gambar, Excel, CSV, teks).

### 3. Modul Pemrosesan
Terdiri dari dua sub-modul utama:
- **Ekstraksi Ciri**: Meliputi metode Histogram, GLCM, Haralick, Hu Moments, LBP, dan HOG untuk menganalisis warna, tekstur, dan bentuk pada gambar.
- **Deteksi Objek**: Meliputi deteksi wajah, mata, warna dominan, kontur, garis, dan sudut pada gambar.

### 4. Modul Tampilan Output
- **Tampilan Gambar**: Menampilkan gambar asli dan hasil pemrosesan di antarmuka aplikasi.
- **Hasil Teks**: Menampilkan hasil numerik dan deskriptif dari ekstraksi ciri dan deteksi objek.
- **Anotasi Visual**: Menampilkan hasil deteksi secara visual pada gambar (misal: kotak pada wajah, garis, titik sudut, dsb).

# Alur Data Sistem

1. **Akuisisi Gambar**: Pengguna memuat gambar dari file atau kamera.
2. **Pemrosesan Awal**: Gambar dikonversi ke format RGB dan disimpan.
3. **Pemilihan Fitur**: Pengguna memilih ekstraksi ciri atau deteksi objek.
4. **Pemrosesan**: Algoritma yang dipilih dijalankan pada gambar.
5. **Tampilan Hasil**: Hasil ditampilkan secara visual dan/atau teks.
6. **Penyimpanan Opsional**: Pengguna dapat menyimpan hasil pemrosesan atau data yang diekstrak.

# Implementasi Teknis

Aplikasi ini dibangun menggunakan:
- **PyQt5**: Untuk antarmuka pengguna grafis
- **OpenCV (cv2)**: Untuk operasi pengolahan citra
- **NumPy**: Untuk operasi numerik pada data gambar
- **Pandas**: Untuk manipulasi dan ekspor data
- **Matplotlib**: Untuk plotting dan visualisasi
- **scikit-image**: Untuk fitur pengolahan citra lanjutan
- **mahotas**: Untuk analisis tekstur (fitur Haralick)
- **scikit-learn**: Untuk operasi machine learning (K-means clustering)

# Kebutuhan Sistem

- Python 3.6 atau lebih baru
- Library: PyQt5, OpenCV, NumPy, Pandas, Matplotlib, scikit-image, mahotas, scikit-learn
- Kamera (untuk fitur pengambilan gambar)
- Memori yang cukup untuk pemrosesan citra 