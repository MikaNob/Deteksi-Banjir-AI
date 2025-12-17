import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from xgboost import XGBClassifier
import google.generativeai as genai
import os

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Pusat Komando Banjir AI",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# SETUP API KEY (OTOMATIS & MANUAL)
# ==========================================
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = None

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg", width=150)
    st.header("Konfigurasi Cerdas")
    
    if GEMINI_API_KEY:
        st.success("✅ API Key Terhubung!")
        if st.checkbox("Ganti API Key"):
            GEMINI_API_KEY = st.text_input("Timpa API Key Baru:", type="password")
    else:
        GEMINI_API_KEY = st.text_input("Masukkan Google Gemini API Key:", type="password", help="Dapatkan di aistudio.google.com")
        if not GEMINI_API_KEY:
            st.warning("⚠️ Masukkan API Key agar fitur AI aktif.")

# ==========================================
# 1. FUNGSI: LOAD MODEL XGBOOST
# ==========================================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    try:
        # Coba load model
        model.load_model("model_banjir_xgboost.json")
        return model
    except Exception as e:
        # JIKA GAGAL, JANGAN DIAM SAJA. TAMPILKAN PENYEBABNYA!
        st.error(f"Terjadi Error Saat Memuat Model: {e}")
        
        # Cek apakah file benar-benar ada di folder server?
        import os
        files = os.listdir('.')
        st.warning(f"Daftar file yang ditemukan di server: {files}")
        return None

# ==========================================
# 2. FUNGSI: GEMINI ADVISOR (AUTO-DETECT)
# ==========================================
def get_gemini_advice(risk_percent, rr, wind, humidity, status_level):
    """
    Otomatis mencari model Gemini yang tersedia di akun pengguna.
    """
    if not GEMINI_API_KEY:
        return None

    genai.configure(api_key=GEMINI_API_KEY)
    
    active_model = None
    
    try:
        # 1. Cari model secara dinamis dari API
        # Kita cari model yang support 'generateContent' dan namanya mengandung 'gemini'
        available_models = list(genai.list_models())
        
        # Prioritas 1: Cari yang ada kata 'flash' (biasanya paling cepat/gratis)
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                active_model = m.name
                break
        
        # Prioritas 2: Jika tidak ada flash, cari yang 'pro'
        if not active_model:
            for m in available_models:
                if 'generateContent' in m.supported_generation_methods and 'pro' in m.name:
                    active_model = m.name
                    break
        
        # Prioritas 3: Apapun yang ada kata 'gemini'
        if not active_model:
            for m in available_models:
                if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name:
                    active_model = m.name
                    break
                    
        if not active_model:
            return "⚠️ Akun API Key ini tidak memiliki akses ke model Gemini apapun. Cek billing/akses di Google AI Studio."

    except Exception as e:
        return f"⚠️ Gagal menghubungi Google AI: {str(e)}"

    # 2. Kirim Prompt menggunakan model yang ditemukan
    try:
        model = genai.GenerativeModel(active_model)
        
        prompt = f"""
        Anda adalah Ahli Manajemen Bencana dan Mitigasi Banjir untuk Kota Semarang.
        
        DATA REAL-TIME:
        - Tingkat Risiko Banjir (AI Prediction): {risk_percent}%
        - Status Level: {status_level}
        - Curah Hujan Akumulasi: {rr} mm (Sangat penting!)
        - Kecepatan Angin: {wind} m/s
        - Kelembapan: {humidity}%

        TUGAS:
        Berikan laporan peringatan dini singkat namun tegas kepada warga.
        Gunakan format berikut:
        1. **Analisis Situasi**: Jelaskan kenapa angka hujan/angin tersebut berbahaya/aman dalam 1 kalimat.
        2. **Peringatan Dini**: Apa yang akan terjadi dalam 1-3 jam ke depan?
        3. **5 Langkah Konkret Warga**: Apa yang harus dilakukan SEKARANG JUGA? (Contoh: Matikan listrik, evakuasi, dll).
        
        Gunakan bahasa Indonesia yang mendesak, empatik, dan mudah dipahami. Gunakan emoji yang sesuai.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error pada model {active_model}: {str(e)}"

# ==========================================
# 3. FUNGSI: LOGIKA STATIC (FALLBACK)
# ==========================================
def get_status_info_static(risk_percent):
    if risk_percent < 40:
        return "AMAN", "green", ["✅ Tetap pantau info BMKG.", "✅ Jaga kebersihan lingkungan."]
    elif 40 <= risk_percent < 75:
        return "WASPADA", "orange", ["⚠️ Bersihkan saluran air.", "⚠️ Siapkan Tas Siaga Bencana.", "⚠️ Amankan elektronik."]
    else:
        return "BAHAYA (AWAS)", "red", ["🚨 EVAKUASI MANDIRI SEKARANG!", "🚨 Matikan listrik dari MCB.", "🚨 Hubungi SAR/BPBD."]

# ==========================================
# 4. FUNGSI: PROSES DATA (CLEANING)
# ==========================================
def proses_data(df_input, is_manual=False):
    df_hasil = df_input.copy()
    
    if not is_manual:
        cols = ['RR', 'RH_avg', 'ff_x', 'ff_avg']
        for c in cols:
            if c in df_hasil.columns:
                df_hasil[c] = pd.to_numeric(df_hasil[c], errors='coerce')
        if 'RR' in df_hasil.columns:
            df_hasil['RR'] = df_hasil['RR'].replace(8888, 0)

    if 'date' in df_hasil.columns:
        df_hasil['date'] = pd.to_datetime(df_hasil['date'])
        df_hasil['bulan'] = df_hasil['date'].dt.month
    else:
        df_hasil['bulan'] = 0

    if not is_manual and 'RR' in df_hasil.columns:
        df_hasil['RR_kemarin'] = df_hasil['RR'].shift(1)
        df_hasil['RR_akumulasi_3hari'] = df_hasil['RR'].rolling(window=3).sum()
        df_hasil = df_hasil.dropna()

    return df_hasil

# ==========================================
# TAMPILAN UTAMA (UI)
# ==========================================
st.title("🌊 Sistem Peringatan Dini & Mitigasi Banjir")
st.markdown("### Powered by XGBoost + Google Gemini AI")

model = load_model()

if model is None:
    st.error("⚠️ Model AI tidak ditemukan. Jalankan 'latih_ai.py' dulu!")
else:
    tab1, tab2 = st.tabs(["📁 Analisis Data Masal (CSV)", "✍️ Cek Cepat & Tanya Gemini"])

    # -------------------------------------------------------------------------
    # TAB 1: CSV & HEATMAP
    # -------------------------------------------------------------------------
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV Data Cuaca", type=['csv'])
        if uploaded_file is not None:
            try:
                df_csv = pd.read_csv(uploaded_file)
                if st.button("🔍 Analisis Risiko"):
                    with st.spinner('XGBoost sedang menghitung probabilitas...'):
                        df_ready = proses_data(df_csv)
                        
                        if df_ready.empty:
                            st.error("Data tidak cukup (perlu min. 3 baris).")
                        else:
                            fitur = ['ff_avg', 'ff_x', 'RH_avg', 'RR', 'RR_kemarin', 'RR_akumulasi_3hari', 'bulan']
                            if not all(col in df_ready.columns for col in fitur):
                                st.error(f"Kolom kurang: {fitur}")
                            else:
                                proba = model.predict_proba(df_ready[fitur])[:, 1]
                                df_ready['Risiko (%)'] = (proba * 100).round(1)

                                # HEATMAP
                                st.subheader("🗺️ Peta Risiko (Simulasi Semarang)")
                                lat_semarang, lon_semarang = -6.9932, 110.4203
                                np.random.seed(42)
                                df_ready['lat'] = lat_semarang + np.random.normal(0, 0.02, len(df_ready))
                                df_ready['lon'] = lon_semarang + np.random.normal(0, 0.02, len(df_ready))
                                df_ready['risk_score'] = df_ready['Risiko (%)']

                                layer = pdk.Layer(
                                    "HeatmapLayer",
                                    data=df_ready,
                                    get_position='[lon, lat]',
                                    get_weight='risk_score',
                                    radius_pixels=60, intensity=1, threshold=0.3
                                )
                                view = pdk.ViewState(latitude=lat_semarang, longitude=lon_semarang, zoom=11)
                                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "Risiko: {risk_score}%"}))

                                # DATA TERAKHIR
                                last_row = df_ready.iloc[-1]
                                risk_val = last_row['Risiko (%)']
                                level_txt, color_txt, _ = get_status_info_static(risk_val)

                                st.divider()
                                st.subheader(f"📅 Status Terkini: {last_row['date'].strftime('%d-%m-%Y')}")
                                
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    st.metric("Probabilitas Banjir", f"{risk_val}%")
                                    st.markdown(f"<h2 style='color:{color_txt}'>{level_txt}</h2>", unsafe_allow_html=True)
                                
                                with c2:
                                    st.write("🤖 **Analisis Asisten Cerdas:**")
                                    if GEMINI_API_KEY:
                                        with st.spinner("Gemini sedang mencari model yang cocok di akun Anda..."):
                                            gemini_response = get_gemini_advice(
                                                risk_val, 
                                                last_row['RR_akumulasi_3hari'], 
                                                last_row['ff_x'], 
                                                last_row['RH_avg'],
                                                level_txt
                                            )
                                            st.markdown(gemini_response)
                                    else:
                                        st.warning("Masukkan API Key di Sidebar untuk analisis detil.")
                                        for act in get_status_info_static(risk_val)[2]:
                                            st.write(act)

            except Exception as e:
                st.error(f"Error: {e}")

    # -------------------------------------------------------------------------
    # TAB 2: MANUAL INPUT & GEMINI
    # -------------------------------------------------------------------------
    with tab2:
        st.write("Simulasi kondisi cuaca untuk mendapatkan saran langsung dari AI.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            ff_x_in = st.number_input("Angin Max (m/s)", 0.0, 50.0, 10.0)
            ff_avg_in = st.number_input("Angin Rata2 (m/s)", 0.0, 50.0, 5.0)
        with c2:
            rh_avg_in = st.number_input("Kelembapan (%)", 0.0, 100.0, 90.0)
            rr_in = st.number_input("Hujan HARI INI (mm)", 0.0, 500.0, 100.0)
        with c3:
            rr_kemarin_in = st.number_input("Hujan KEMARIN (mm)", 0.0, 500.0, 50.0)
            rr_lusa_in = st.number_input("Hujan 2 HARI LALU (mm)", 0.0, 500.0, 20.0)

        if st.button("⚡ ANALISIS DENGAN GEMINI"):
            akumulasi = rr_in + rr_kemarin_in + rr_lusa_in
            
            data_manual = pd.DataFrame({
                'date': [pd.Timestamp.now()],
                'ff_avg': [ff_avg_in], 'ff_x': [ff_x_in], 'RH_avg': [rh_avg_in],
                'RR': [rr_in], 'RR_kemarin': [rr_kemarin_in], 'RR_akumulasi_3hari': [akumulasi]
            })
            data_manual = proses_data(data_manual, is_manual=True)
            
            fitur = ['ff_avg', 'ff_x', 'RH_avg', 'RR', 'RR_kemarin', 'RR_akumulasi_3hari', 'bulan']
            risk_percent = (model.predict_proba(data_manual[fitur])[0, 1] * 100).round(1)
            
            level_txt, color_txt, static_advice = get_status_info_static(risk_percent)

            st.divider()
            st.markdown(f"<h1 style='text-align: center; color: {color_txt};'>RISIKO BANJIR: {risk_percent}% ({level_txt})</h1>", unsafe_allow_html=True)
            
            st.subheader("🤖 Rekomendasi Mitigasi & Evakuasi (Gemini AI)")
            
            if GEMINI_API_KEY:
                with st.spinner("Sedang berkonsultasi dengan ahli mitigasi digital..."):
                    advice = get_gemini_advice(risk_percent, akumulasi, ff_x_in, rh_avg_in, level_txt)
                    st.success("Laporan diterima.")
                    st.markdown(advice)
            else:
                st.warning("⚠️ Masukkan API Key untuk saran cerdas.")
                for act in static_advice:

                    st.write(act)
