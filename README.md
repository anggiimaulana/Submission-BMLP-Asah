# Submission Akhir BMLP: Clustering dan Klasifikasi Transaksi Finansial

Repositori ini berisi proyek akhir Dicoding program Belajar Machine Learning (BMLP) oleh **Anggi Maulana**. Proyek mencakup alur *end-to-end* machine learning: *unsupervised learning* (Clustering) untuk segmentasi nasabah, dilanjutkan *supervised learning* (Klasifikasi) untuk memprediksi label cluster.

---

## Struktur File

```text
.
├── [Clustering]_Submission_Akhir_BMLP_Anggi_Maulana.ipynb    # Notebook clustering & analisis segmen
├── [Klasifikasi]_Submission_Akhir_BMLP_Anggi_Maulana.ipynb   # Notebook klasifikasi & tuning model
├── data_clustering.csv                                        # Data hasil clustering (skala standar)
├── data_clustering_inverse.csv                                # Data hasil clustering (nilai riil/inverse)
├── model_clustering.h5                                        # Model KMeans utama
├── PCA_model_clustering.h5                                    # Model KMeans berbasis fitur PCA
├── decision_tree_model.h5                                     # Model Decision Tree classifier
├── explore_RandomForest_classification.h5                     # Model Random Forest classifier baseline
└── tuning_classification.h5                                   # Model Random Forest hasil GridSearchCV
```

---

## Ringkasan Alur & Metodologi

### 1. Clustering (Unsupervised Learning)
- **Dataset**: Data transaksi finansial (2.512 entri awal, 1.945 data setelah pembersihan *outlier* dan duplikasi).
- **Preprocessing**: Imputasi/pembersihan nilai hilang, *outlier handling*, agregasi *TransactionGroup*, *StandardScaler*, dan reduksi dimensi dengan PCA.
- **Modeling**: K-Means Clustering dengan penentuan jumlah cluster optimal menggunakan metode *Silhouette Score Elbow*.
- **Evaluasi**: Silhouette Score optimal pada $k = 2$ bernilai **0.572**.
- **Segmentasi**:
  - **Cluster 0 (Nasabah Mapan & Konservatif)**: Saldo rata-rata tinggi (5.142,17), nominal transaksi rata-rata 255,55, dominan profesi *Doctor*, lokasi dominan Charlotte.
  - **Cluster 1 (Nasabah Muda & Transaktif)**: Saldo rata-rata lebih rendah (5.058,81), nominal transaksi rata-rata 258,15 dengan frekuensi *High*, dominan profesi *Student*, lokasi dominan Tucson.

### 2. Klasifikasi (Supervised Learning)
- **Dataset Input**: `data_clustering_inverse.csv` menggunakan kolom `Target` hasil clustering.
- **Preprocessing**: One-Hot Encoding pada fitur kategorikal (`pd.get_dummies`), pemisahan data train-test 80:20 dengan *stratified split*.
- **Model**:
  - **Decision Tree**: Model dasar (*baseline*).
  - **Random Forest**: Model lanjutan ensemble (*Skilled*).
  - **Tuned Random Forest**: Optimasi hyperparameter dengan `GridSearchCV` 5-fold cross-validation (*Advanced*).
- **Evaluasi**: Seluruh metrik (Accuracy, Precision, Recall, F1-Score) mencapai **1.00 (100%)** pada data pengujian.

---

## Kebutuhan Sistem & Instalasi

Pastikan Python 3.8+ terpasang, lalu instal paket berikut:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn yellowbrick joblib
```

## Menjalankan Proyek

1. Jalankan `[Clustering]_Submission_Akhir_BMLP_Anggi_Maulana.ipynb` untuk menjalankan data pipeline, menghasilkan visualisasi, serta mengekspor data dan model clustering.
2. Jalankan `[Klasifikasi]_Submission_Akhir_BMLP_Anggi_Maulana.ipynb` untuk melatih dan mengevaluasi model klasifikasi berdasarkan data hasil clustering.
