# ==========================================
# STUDI KASUS 6: Atribusi Pemicu Deforestasi dengan AI & Citra Satelit
# Berdasarkan Materi Praktisi oleh Dr. Bart Slaughter (Wageningen University & Research)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- Memulai Analisis Atribusi Pemicu Deforestasi Berbasis AI ---")

# 1. Simulasi Dataset Peringatan Deforestasi & Multi-Sensor Features
np.random.seed(42)
n_alerts = 1000

deforestation_data = pd.DataFrame({
    'alert_id': [f"ALT-{i:04d}" for i in range(1, n_alerts + 1)],
    'radar_backscatter_change': np.random.uniform(-15.0, -2.0, size=n_alerts), # Perubahan sinyal radar Sentinel-1 (dB)
    'ndvi_optical_drop': np.random.uniform(0.1, 0.8, size=n_alerts),          # Penurunan indeks vegetasi Sentinel-2
    'patch_size_hectares': np.random.exponential(scale=5.0, size=n_alerts),   # Luas area deforestasi (hektar)
    'proximity_to_road_km': np.random.exponential(scale=2.0, size=n_alerts)   # Jarak ke jalan terdekat
})

# 2. Simulasi Atribusi Label Pemicu Langsung (Direct Drivers)
# Driver: 0 = Pertanian (Agriculture), 1 = Pertambangan (Mining), 2 = Pembalakan (Logging), 3 = Kebakaran (Wildfire)
conditions = [
    (deforestation_data['patch_size_hectares'] > 10) & (deforestation_data['ndvi_optical_drop'] > 0.5),
    (deforestation_data['radar_backscatter_change'] < -10) & (deforestation_data['proximity_to_road_km'] < 1.5),
    (deforestation_data['patch_size_hectares'] <= 3) & (deforestation_data['proximity_to_road_km'] < 3.0),
    (deforestation_data['radar_backscatter_change'] >= -5)
]
choices = [0, 1, 2, 3]
deforestation_data['driver_label'] = np.select(conditions, choices, default=0)

print(f"Total Peringatan Deforestasi Tercatat: {n_alerts} record")
print("Distribusi Pemicu Deforestasi (0: Pertanian, 1: Tambang, 2: Logging, 3: Kebakaran):\n", 
      deforestation_data['driver_label'].value_counts())

# 3. Persiapan Data untuk Model Klasifikasi (X dan y)
X = deforestation_data[['radar_backscatter_change', 'ndvi_optical_drop', 'patch_size_hectares', 'proximity_to_road_km']]
y = deforestation_data['driver_label']

# Data Splitting (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training Model Klasifikasi (Random Forest sebagai Benchmark AI)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("\n[SUKSES] Model Klasifikasi Pemicu Deforestasi berhasil dilatih.")

# 5. Evaluasi Performa Model
y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Atribusi Model: {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Pertanian', 'Pertambangan', 'Logging', 'Kebakaran']))

# 6. Ekspor Hasil Analisis Atribusi untuk Kebutuhan Monitoring Cloud
output_report = X_test.copy()
output_report['actual_driver'] = y_test
output_report['predicted_driver'] = y_pred
output_report.to_csv('deforestation_attribution_output.csv', index=False)
print("\n[SUKSES] Laporan atribuasi berhasil diekspor ke 'deforestation_attribution_output.csv'.")