import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import skimage.feature as skfeature
from sklearn.cluster import KMeans
from collections import Counter
import mahotas as mt

class ImageProcessingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI file
        uic.loadUi('gui.ui', self)
        
        # Variabel untuk menyimpan gambar
        self.original_image = None
        self.processed_image = None
        self.second_image = None
        self.extracted_features = None
        
        # Inisialisasi area untuk menampilkan hasil ekstraksi ciri
        self.text_features = QtWidgets.QTextEdit(self.ekstrak_ciri)
        self.text_features.setGeometry(10, 10, 231, 521)
        self.text_features.setFont(QtGui.QFont("Courier New", 10))
        
        # Inisialisasi area untuk menampilkan hasil deteksi objek
        self.detection_label = QtWidgets.QLabel(self.Objek_Foto)
        self.detection_label.setGeometry(10, 10, 291, 271)
        self.detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detection_label.setStyleSheet("border: 1px solid black;")
        
        # Inisialisasi area untuk menampilkan informasi deteksi
        self.text_detection = QtWidgets.QTextEdit(self.output_hasil)
        self.text_detection.setGeometry(10, 10, 291, 101)
        self.text_detection.setFont(QtGui.QFont("Courier New", 10))
        
        # Connect signals
        self.pushButton.clicked.connect(self.show_feature_detection_tab)
        self.pushButton_2.clicked.connect(self.extract_features)
        self.pushButton_3.clicked.connect(self.detect_objects)
        self.pushButton_4.clicked.connect(self.save_features)
        self.pushButton_5.clicked.connect(self.open_image)
        self.pushButton_6.clicked.connect(self.capture_from_camera)
        self.pushButton_7.clicked.connect(self.save_image)
        self.pushButton_8.clicked.connect(self.export_pixel_data)
        
        # Set window title
        self.setWindowTitle("Aplikasi Pengolahan Citra Digital")
        
    def show_feature_detection_tab(self):
        # Fungsi ini bisa digunakan untuk menampilkan tab ekstraksi ciri & deteksi objek
        # jika aplikasi memiliki multiple tab
        pass
        
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Buka Gambar", "", 
            "Gambar (*.png *.jpg *.jpeg *.bmp *.tif);;Semua File (*)"
        )
        
        if file_path:
            try:
                # Baca gambar dengan OpenCV
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.processed_image = self.original_image.copy()
                
                # Tampilkan gambar
                self.display_image(self.original_image, self.detection_label)
                
                QMessageBox.information(self, "Info", "Gambar berhasil dibuka!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal membuka gambar: {str(e)}")

    def capture_from_camera(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                QMessageBox.critical(self, "Error", "Tidak dapat mengakses kamera!")
                return
                
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.processed_image = self.original_image.copy()
                
                # Tampilkan gambar
                self.display_image(self.original_image, self.detection_label)
                
                QMessageBox.information(self, "Info", "Gambar berhasil diambil dari kamera!")
            else:
                QMessageBox.critical(self, "Error", "Gagal mengambil gambar dari kamera!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat mengakses kamera: {str(e)}")

    def display_image(self, image, label):
        if image is None:
            return
            
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio)
        
        label.setPixmap(pixmap)

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada gambar untuk disimpan!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Gambar", "", 
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, "Info", "Gambar berhasil disimpan!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan gambar: {str(e)}")

    def export_pixel_data(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada data piksel untuk diekspor!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Data Piksel", "", 
            "Excel Files (*.xlsx);;Text Files (*.txt);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Menyiapkan data piksel asli
                if len(self.original_image.shape) == 3:  # RGB
                    original_data = self.original_image.reshape(-1, 3)
                    original_df = pd.DataFrame(original_data, columns=["R_original", "G_original", "B_original"])
                else:  # Grayscale
                    original_data = self.original_image.reshape(-1)
                    original_df = pd.DataFrame(original_data, columns=["Gray_original"])
                
                # Menyiapkan data piksel hasil pemrosesan
                if len(self.processed_image.shape) == 3:  # RGB
                    processed_data = self.processed_image.reshape(-1, 3)
                    processed_df = pd.DataFrame(processed_data, columns=["R_processed", "G_processed", "B_processed"])
                else:  # Grayscale
                    processed_data = self.processed_image.reshape(-1)
                    processed_df = pd.DataFrame(processed_data, columns=["Gray_processed"])
                
                # Menggabungkan kedua DataFrame
                result_df = pd.concat([original_df, processed_df], axis=1)
                
                # Menyimpan data sesuai format
                if file_path.endswith(".xlsx"):
                    result_df.to_excel(file_path, index=False)
                elif file_path.endswith(".txt"):
                    result_df.to_csv(file_path, sep='\t', index=False)
                else:
                    result_df.to_csv(file_path, index=False)
                    
                QMessageBox.information(self, "Info", "Data piksel berhasil diekspor!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal mengekspor data piksel: {str(e)}")

    def extract_features(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan buka gambar terlebih dahulu!")
            return
            
        feature_type = self.comboBox.currentText()
        
        try:
            # Pastikan gambar dalam grayscale untuk beberapa tipe ekstraksi ciri
            if len(self.original_image.shape) == 3:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.original_image.copy()
                
            self.text_features.clear()
            self.text_features.append(f"Ekstraksi Ciri: {feature_type}\n\n")
            
            # Histogram
            if feature_type == "Histogram":
                if len(self.original_image.shape) == 3:  # RGB
                    hist_features = []
                    for i in range(3):
                        hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        hist_features.extend(hist[:10])  # Ambil 10 bins pertama
                        
                    self.text_features.append("Histogram RGB (10 bins pertama per channel):\n")
                    for i, value in enumerate(hist_features):
                        channel = "R" if i < 10 else "G" if i < 20 else "B"
                        bin_num = i % 10
                        self.text_features.append(f"{channel} Bin {bin_num}: {value:.4f}\n")
                        
                else:  # Grayscale
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    
                    self.text_features.append("Histogram Grayscale (20 bins pertama):\n")
                    for i in range(20):
                        self.text_features.append(f"Bin {i}: {hist[i]:.4f}\n")
            
            # GLCM (Gray Level Co-occurrence Matrix)
            elif feature_type == "GLCM":
                # Gunakan skimage untuk GLCM
                glcm = skfeature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                
                # Ekstrak properti GLCM
                contrast = skfeature.graycoprops(glcm, 'contrast')
                dissimilarity = skfeature.graycoprops(glcm, 'dissimilarity')
                homogeneity = skfeature.graycoprops(glcm, 'homogeneity')
                energy = skfeature.graycoprops(glcm, 'energy')
                correlation = skfeature.graycoprops(glcm, 'correlation')
                
                self.text_features.append("GLCM Properties:\n")
                self.text_features.append(f"Contrast: {contrast[0][0]:.4f}, {contrast[0][1]:.4f}, {contrast[0][2]:.4f}, {contrast[0][3]:.4f}\n")
                self.text_features.append(f"Dissimilarity: {dissimilarity[0][0]:.4f}, {dissimilarity[0][1]:.4f}, {dissimilarity[0][2]:.4f}, {dissimilarity[0][3]:.4f}\n")
                self.text_features.append(f"Homogeneity: {homogeneity[0][0]:.4f}, {homogeneity[0][1]:.4f}, {homogeneity[0][2]:.4f}, {homogeneity[0][3]:.4f}\n")
                self.text_features.append(f"Energy: {energy[0][0]:.4f}, {energy[0][1]:.4f}, {energy[0][2]:.4f}, {energy[0][3]:.4f}\n")
                self.text_features.append(f"Correlation: {correlation[0][0]:.4f}, {correlation[0][1]:.4f}, {correlation[0][2]:.4f}, {correlation[0][3]:.4f}\n")
                
            # Haralick Features
            elif feature_type == "Haralick":
                # Gunakan mahotas untuk Haralick Features
                haralick_features = mt.features.haralick(gray)
                mean_haralick = np.mean(haralick_features, axis=0)
                
                feature_names = [
                    "Angular Second Moment", "Contrast", "Correlation", "Sum of Squares: Variance",
                    "Inverse Difference Moment", "Sum Average", "Sum Variance", "Sum Entropy",
                    "Entropy", "Difference Variance", "Difference Entropy",
                    "Information Measure of Correlation 1", "Information Measure of Correlation 2"
                ]
                
                self.text_features.append("Haralick Features:\n")
                for i, feature in enumerate(mean_haralick):
                    self.text_features.append(f"{feature_names[i]}: {feature:.4f}\n")
                    
            # Hu Moments
            elif feature_type == "Hu Moments":
                moments = cv2.moments(gray)
                hu_moments = cv2.HuMoments(moments)
                
                self.text_features.append("Hu Moments:\n")
                for i, moment in enumerate(hu_moments.flatten()):
                    # Log transform untuk membuat nilai lebih mudah dibaca
                    log_moment = -np.sign(moment) * np.log10(abs(moment))
                    self.text_features.append(f"Moment {i+1}: {log_moment:.4f}\n")
                    
            # Local Binary Pattern (LBP)
            elif feature_type == "LBP":
                radius = 3
                n_points = 8 * radius
                lbp = skfeature.local_binary_pattern(gray, n_points, radius, method="uniform")
                
                # Histogram dari LBP
                n_bins = int(n_points + 2)
                hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
                
                self.text_features.append(f"LBP Histogram (radius={radius}, points={n_points}):\n")
                for i, value in enumerate(hist):
                    self.text_features.append(f"Bin {i}: {value:.4f}\n")
                    
            # Histogram of Oriented Gradients (HOG)
            elif feature_type == "HOG":
                # Resize gambar untuk HOG
                resized = cv2.resize(gray, (64, 128))
                
                # Compute HOG
                hog_features, hog_image = skfeature.hog(
                    resized, 
                    orientations=9, 
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), 
                    visualize=True, 
                    block_norm='L2-Hys'
                )
                
                self.text_features.append(f"HOG Features (total {len(hog_features)} features):\n")
                # Tampilkan beberapa nilai HOG
                for i in range(min(20, len(hog_features))):
                    self.text_features.append(f"Feature {i}: {hog_features[i]:.4f}\n")
                
                # Tampilkan gambar HOG
                self.processed_image = (hog_image * 255).astype(np.uint8)
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
                self.display_image(self.processed_image, self.detection_label)
                
            # Simpan hasil ekstraksi sebagai atribut
            self.extracted_features = {
                'type': feature_type,
                'data': self.text_features.toPlainText()
            }
            
            QMessageBox.information(self, "Info", f"Ekstraksi ciri {feature_type} berhasil dilakukan!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal melakukan ekstraksi ciri: {str(e)}")
            
    def save_features(self):
        if not hasattr(self, 'extracted_features') or self.extracted_features is None:
            QMessageBox.warning(self, "Peringatan", "Belum ada fitur yang diekstrak!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil Ekstraksi Ciri", "", 
            "Text Files (*.txt);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.extracted_features['data'])
                QMessageBox.information(self, "Info", "Hasil ekstraksi ciri berhasil disimpan!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan hasil ekstraksi: {str(e)}")

    def detect_objects(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan buka gambar terlebih dahulu!")
            return
            
        detection_type = self.comboBox_2.currentText()
        
        try:
            # Buat salinan gambar untuk ditampilkan hasil deteksi
            detection_image = self.original_image.copy()
            
            # Hapus teks sebelumnya
            self.text_detection.clear()
            
            # Deteksi Wajah menggunakan Haar Cascade
            if detection_type == "Deteksi wajah (Haar Cascade)":
                # Konversi ke grayscale
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                
                # Load Haar Cascade untuk deteksi wajah
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                # Deteksi wajah
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Gambar kotak di sekitar wajah
                for (x, y, w, h) in faces:
                    cv2.rectangle(detection_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Tampilkan informasi
                self.text_detection.append(f"Terdeteksi {len(faces)} wajah\n")
                for i, (x, y, w, h) in enumerate(faces):
                    self.text_detection.append(f"Wajah {i+1}: Posisi=({x},{y}), Ukuran={w}x{h}\n")
            
            # Deteksi Mata
            elif detection_type == "Deteksi Mata":
                # Konversi ke grayscale
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                
                # Load Haar Cascade untuk deteksi mata
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                
                # Deteksi mata
                eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Gambar kotak di sekitar mata
                for (x, y, w, h) in eyes:
                    cv2.rectangle(detection_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Tampilkan informasi
                self.text_detection.append(f"Terdeteksi {len(eyes)} mata\n")
                for i, (x, y, w, h) in enumerate(eyes):
                    self.text_detection.append(f"Mata {i+1}: Posisi=({x},{y}), Ukuran={w}x{h}\n")
            
            # Deteksi Warna Dominan
            elif detection_type == "Deteksi Warna Dominan":
                # Reshape gambar untuk KMeans
                pixels = self.original_image.reshape(-1, 3)
                
                # Konversi ke float untuk perhitungan
                pixels = np.float32(pixels)
                
                # Tentukan jumlah cluster (warna dominan)
                n_colors = 5
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
                flags = cv2.KMEANS_RANDOM_CENTERS
                
                # Lakukan clustering
                _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                
                # Konversi kembali ke uint8
                centers = np.uint8(centers)
                
                # Hitung frekuensi setiap cluster
                counter = Counter(labels.flatten())
                
                # Urutkan warna dominan berdasarkan frekuensi
                ordered_colors = sorted([(centers[i], counter[i]) for i in counter.keys()], 
                                       key=lambda x: x[1], reverse=True)
                
                # Tampilkan informasi
                self.text_detection.append("Warna Dominan (RGB):\n")
                
                # Buat gambar dengan warna dominan
                bar_width = detection_image.shape[1] // n_colors
                color_bar = np.zeros((50, detection_image.shape[1], 3), dtype=np.uint8)
                
                for i, ((b, g, r), count) in enumerate(ordered_colors):
                    percentage = count / len(labels.flatten()) * 100
                    self.text_detection.append(f"Warna {i+1}: RGB=({r},{g},{b}), {percentage:.2f}%\n")
                    
                    # Tambahkan ke color bar
                    start = i * bar_width
                    end = (i + 1) * bar_width if i < n_colors - 1 else detection_image.shape[1]
                    color_bar[:, start:end] = [r, g, b]
                
                # Tambahkan color bar ke gambar
                detection_image = np.vstack([detection_image, color_bar])
            
            # Deteksi Kontur
            elif detection_type == "Deteksi Kontur":
                # Konversi ke grayscale
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                
                # Threshold untuk mendapatkan gambar biner
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                
                # Cari kontur
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Gambar kontur
                cv2.drawContours(detection_image, contours, -1, (0, 255, 0), 2)
                
                # Tampilkan informasi
                self.text_detection.append(f"Terdeteksi {len(contours)} kontur\n")
                
                # Tampilkan informasi detail untuk 5 kontur terbesar
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                for i, contour in enumerate(sorted_contours):
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    self.text_detection.append(f"Kontur {i+1}: Luas={area:.2f}, Keliling={perimeter:.2f}\n")
            
            # Deteksi Garis (Hough Line Transform)
            elif detection_type == "Deteksi Garis":
                # Konversi ke grayscale
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                
                # Deteksi tepi dengan Canny
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                
                # Deteksi garis dengan Hough Line Transform
                lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
                
                if lines is not None:
                    # Gambar garis
                    for i, line in enumerate(lines[:20]):  # Batasi menjadi 20 garis saja
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(detection_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Tampilkan informasi
                    self.text_detection.append(f"Terdeteksi {len(lines)} garis\n")
                    self.text_detection.append(f"Menampilkan 20 garis pertama\n")
                else:
                    self.text_detection.append("Tidak ada garis yang terdeteksi\n")
            
            # Deteksi Sudut (Harris Corner Detection)
            elif detection_type == "Deteksi Sudut":
                # Konversi ke grayscale
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                
                # Deteksi sudut dengan Harris Corner Detection
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                
                # Dilasi untuk menandai sudut
                dst = cv2.dilate(dst, None)
                
                # Threshold untuk menentukan sudut
                detection_image[dst > 0.01 * dst.max()] = [0, 0, 255]
                
                # Hitung jumlah sudut
                corners = np.sum(dst > 0.01 * dst.max())
                
                # Tampilkan informasi
                self.text_detection.append(f"Terdeteksi {corners} sudut\n")
            
            # Tampilkan gambar hasil deteksi
            self.processed_image = detection_image
            self.display_image(detection_image, self.detection_label)
            
            QMessageBox.information(self, "Info", f"Deteksi objek {detection_type} berhasil dilakukan!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal melakukan deteksi objek: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_()) 