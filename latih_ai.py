import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # <--- [UBAHAN 1] Tambahkan library ini untuk menyimpan model sebagai .pkl
import sys

# -----------------------------------------------------------------------------
# 1. INSTALASI LIBRARY (Cek XGBoost)
# -----------------------------------------------------------------------------
try:
    import xgboost
except ImportError:
    print("Menginstall XGBoost...")

# -----------------------------------------------------------------------------
# 2. LOAD DATA DARI GITHUB
# -----------------------------------------------------------------------------
url_data = 'https://raw.githubusercontent.com/MikaNob/DATAAI/refs/heads/main/data_finish.csv'

print("Sedang mengambil data dari GitHub...")
try:
    df = pd.read_csv(url_data)
    print("✅ Data berhasil dimuat!")
    print(f"Jumlah baris data: {len(df)}")
except Exception as e:
    print("❌ Gagal memuat data.")
    print(f"Error: {e}")
    sys.exit()

# -----------------------------------------------------------------------------
# 3. PEMBERSIHAN DATA (CLEANING)
# -----------------------------------------------------------------------------
print("\nMelakukan pembersihan data...")

# A. Konversi Tanggal
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# B. Bersihkan Kolom Numerik
cols_to_numeric = ['RR', 'RH_avg', 'ff_x', 'ff_avg', 'flood']
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# C. Penanganan Khusus Curah Hujan (RR)
if 'RR' in df.columns:
    df['RR'] = df['RR'].replace(8888, 0).fillna(0)

# D. HAPUS DATA KOSONG
print(f"Jumlah baris sebelum cleaning: {len(df)}")
df = df.dropna()
print(f"Jumlah baris setelah cleaning: {len(df)}")

# E. Pastikan format target benar
if 'flood' in df.columns:
    df['flood'] = df['flood'].astype(int)

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print("Membuat fitur Time-Series...")

df['bulan'] = df['date'].dt.month
df['RR_kemarin'] = df['RR'].shift(1)
df['RR_akumulasi_3hari'] = df['RR'].rolling(window=3).sum()

df_clean = df.dropna().copy()
print(f"Data siap training: {len(df_clean)} baris.")

# -----------------------------------------------------------------------------
# 5. PERSIAPAN TRAINING
# -----------------------------------------------------------------------------
fitur_final = [
    'ff_avg', 
    'ff_x', 
    'RH_avg', 
    'RR', 
    'RR_kemarin', 
    'RR_akumulasi_3hari', 
    'bulan'
]
target = 'flood'

X = df_clean[fitur_final]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------------------------------------------------------
# 6. TRAINING XGBOOST
# -----------------------------------------------------------------------------
print("\nSedang melatih model XGBoost...")

model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("Training Selesai!")

# -----------------------------------------------------------------------------
# 7. EVALUASI MODEL
# -----------------------------------------------------------------------------
prediksi = model.predict(X_test)
acc = accuracy_score(y_test, prediksi)
print(f"\n=== AKURASI MODEL: {acc * 100:.2f}% ===")

# -----------------------------------------------------------------------------
# 8. SIMPAN MODEL (FORMAT BARU: .pkl)
# -----------------------------------------------------------------------------
# [UBAHAN 2] Menggunakan joblib.dump untuk menyimpan format pickle
nama_file_model = "model_banjir.pkl"
joblib.dump(model, nama_file_model)

print(f"\n✅ BERHASIL! Model disimpan sebagai '{nama_file_model}' (Format Pickle)")
print("👉 Langkah Selanjutnya: Upload file .pkl ini ke GitHub untuk menggantikan file .json lama.")