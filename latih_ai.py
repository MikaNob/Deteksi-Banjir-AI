import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1. INSTALASI LIBRARY (Jika belum ada di environment)
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
    # Membaca CSV
    df = pd.read_csv(url_data)
    print("✅ Data berhasil dimuat!")
    print(f"Jumlah baris data: {len(df)}")
    print("Contoh 5 data teratas:")
    # ✅ Gunakan print() biasa
    print(df.head())
except Exception as e:
    print("❌ Gagal memuat data.")
    print(f"Error: {e}")
    # Stop program jika data gagal load
    raise e

# -----------------------------------------------------------------------------
# 3. PEMBERSIHAN DATA (CLEANING)
# -----------------------------------------------------------------------------
print("\nMelakukan pembersihan data...")

# A. Konversi Tanggal
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# B. Bersihkan Kolom Numerik (RR, RH_avg, ff_x, ff_avg)
cols_to_numeric = ['RR', 'RH_avg', 'ff_x', 'ff_avg', 'flood']

for col in cols_to_numeric:
    if col in df.columns:
        # Ubah ke angka, error jadi NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

# C. Penanganan Khusus Curah Hujan (RR)
# Di data BMKG, kadang ada nilai 8888 (tak terukur). Kita anggap 0 atau data error.
if 'RR' in df.columns:
    df['RR'] = df['RR'].replace(8888, 0)
    df['RR'] = df['RR'].fillna(0) # Isi data kosong dengan 0

# D. HAPUS DATA KOSONG (MODIFIKASI BARU)
# Hapus baris jika ADA SATU SAJA kolom yang kosong (NaN/Null)
print(f"Jumlah baris sebelum cleaning: {len(df)}")
df = df.dropna() 
print(f"Jumlah baris setelah cleaning: {len(df)}")

# E. Pastikan format target benar (Integer)
if 'flood' in df.columns:
    df['flood'] = df['flood'].astype(int)

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (RAHASIA XGBOOST JADI PINTAR)
# -----------------------------------------------------------------------------
print("Membuat fitur Time-Series...")

# A. Fitur Musim (Bulan)
# Penting: Menangkap pola musim hujan (Januari/Februari rawan banjir)
df['bulan'] = df['date'].dt.month

# B. Fitur LAG (Masa Lalu)
# "Apakah kemarin hujan?"
df['RR_kemarin'] = df['RR'].shift(1)

# C. Fitur WINDOW (Akumulasi) - KRUSIAL UNTUK BANJIR
# Menghitung total hujan dalam 3 hari terakhir.
# Banjir sering terjadi karena akumulasi, bukan hanya hujan hari ini.
df['RR_akumulasi_3hari'] = df['RR'].rolling(window=3).sum()

# D. Hapus Data Kosong (NaN)
# Karena proses shift() dan rolling(), baris-baris awal akan kosong. Kita hapus.
df_clean = df.dropna().copy()

print(f"Data siap training: {len(df_clean)} baris.")

# -----------------------------------------------------------------------------
# 5. PERSIAPAN TRAINING
# -----------------------------------------------------------------------------

# Fitur yang dipilih (Sesuai request: Angin, Lembap, Hujan, Tanggal)
# Kita TIDAK memasukkan 'tinggi_air_laut' karena data tidak ada.
fitur_final = [
    'ff_avg',             # Kecepatan Angin Rata-rata
    'ff_x',               # Kecepatan Angin Maksimum (Indikator Badai/CENS)
    'RH_avg',             # Kelembapan Rata-rata
    'RR',                 # Curah Hujan Hari Ini
    'RR_kemarin',         # Curah Hujan Kemarin
    'RR_akumulasi_3hari', # Akumulasi Hujan 3 Hari
    'bulan'               # Penanda Musim
]

target = 'flood'

X = df_clean[fitur_final]
y = df_clean[target]

# Split Data: 80% Latih, 20% Uji
# PENTING: shuffle=False karena ini data deret waktu (Time Series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------------------------------------------------------
# 6. TRAINING XGBOOST
# -----------------------------------------------------------------------------
print("\nSedang melatih model XGBoost...")

model = XGBClassifier(
    n_estimators=100,     # Jumlah pohon keputusan
    learning_rate=0.05,   # Kecepatan belajar
    max_depth=5,          # Kedalaman pohon
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("Training Selesai!")

# -----------------------------------------------------------------------------
# 7. EVALUASI MODEL
# -----------------------------------------------------------------------------
prediksi = model.predict(X_test)

print("\n" + "="*40)
print("HASIL PREDIKSI AI")
print("="*40)
print(f"Akurasi Model: {accuracy_score(y_test, prediksi) * 100:.2f}%")

print("\nLaporan Detail Klasifikasi:")
# Perhatikan nilai 'Recall' untuk kelas '1' (Banjir).
# Recall tinggi = Bagus mendeteksi banjir beneran.
print(classification_report(y_test, prediksi))

# -----------------------------------------------------------------------------
# 8. VISUALISASI HASIL
# -----------------------------------------------------------------------------

# A. Grafik Feature Importance (Faktor Apa yang Paling Berpengaruh?)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='#1f77b4')
plt.title('Faktor Paling Mempengaruhi Prediksi Banjir (Menurut AI)')
plt.xlabel('Tingkat Kepentingan (Score)')
plt.show()

# B. Confusion Matrix (Benar vs Salah Tebak)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, prediksi)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Banjir', 'Banjir'], yticklabels=['Tidak Banjir', 'Banjir'])
plt.ylabel('Kenyataan')
plt.xlabel('Prediksi AI')
plt.title('Confusion Matrix')
plt.show()

# 1. Simpan model ke format JSON
nama_file_model = "model_banjir_xgboost.json"
model.save_model(nama_file_model)

print(f"✅ Model berhasil disimpan sebagai '{nama_file_model}'")