import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SmartDissolve AI", layout="wide")

# --- SARI LACİVERT BAŞLIK TASARIMI ---
st.sidebar.markdown(
    """
    <div style="background-color: #002D72; padding: 15px; border-radius: 10px; border: 3px solid #FFD700;">
        <h1 style="color: #FFD700; margin: 0; font-size: 1.8rem; text-align: center;">💊 SmartDissolve AI</h1>
        <p style="color: white; margin: 0; font-size: 0.8rem; text-align: center;">Predictive Dissolution Suite</p>
    </div>
    """, 
    unsafe_allow_html=True
)
st.sidebar.divider()

# --- ANALYTICAL SUITE (MENÜ) ---
menu = st.sidebar.radio(
    "Analytical Suite",
    ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"]
)

# --- VERİ İŞLEME FONKSİYONU ---
def process_data(file):
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": v.shape[1], "raw": v[mask]}
    except Exception as e:
        st.error(f"Dosya okuma hatası ({file.name}): {e}")
        return None

def calculate_model_independent(t, q):
    dt = np.diff(t, prepend=0)
    de = (np.cumsum(q * dt)[-1] / (t[-1] * 100)) * 100
    mdt = np.sum((t - (dt/2)) * np.diff(q, prepend=0)) / q[-1] if q[-1] > 0 else 0
    return de, mdt

# --- DOSYA YÜKLEME ---
test_files = st.sidebar.file_uploader("Test Verileri (Çoklu Seçebilirsiniz)", type=['xlsx', 'csv'], accept_multiple_files=True)
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

# --- ANA MANTIK ---
if test_files:
    # Tüm dosyaları işle ve bir sözlükte sakla
    data_registry = {}
    for f in test_files:
        processed = process_data(f)
        if processed:
            data_registry[f.name] = processed

    # Kullanıcıya hangi dosyayı analiz etmek istediğini sor
    selected_filename = st.selectbox("Analiz edilecek dosyayı seçin:", list(data_registry.keys()))
    active_data = data_registry[selected_filename]
    
    # Referans verisi varsa işle
    ref_data = process_data(ref_file) if ref_file else None

    # Değişkenleri tanımla
    t_raw, q_raw = active_data["t"], active_data["mean"]
    de, mdt = calculate_model_independent(t_raw, q_raw)

    if menu == "📈 Salım Profilleri":
        st.subheader(f"📊 {selected_filename} - İstatistiksel Özet")
        
        # Grafik Çizimi
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_raw, q_raw, yerr=active_data["std"], fmt='-ok', label=f"Test: {selected_filename}", capsize=5)
        if ref_data:
            ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", capsize=5)
        ax.set_xlabel("Zaman (Dakika)")
        ax.set_ylabel("Salım (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Metrikler
        c1, c2 = st.columns(2)
        c1.metric("Dissolution Efficiency (DE %)", f"{de:.2f}%")
        c2.metric("Mean Dissolution Time (MDT)", f"{mdt:.2f} dk")

    elif menu == "🧮 Kinetik Model Fitting":
        st.subheader("Kinetik Model Fitting Analizi")
        st.info("Bu modülde seçili dosya için 16 farklı model fitting işlemi gerçekleştirilir.")
        # (Model fitting kodları buraya eklenecek)

    elif menu == "📊 f1 & f2 Benzerlik Analizi":
        if ref_data:
            st.subheader("f1 & f2 Benzerlik Analizi")
            # f1 f2 hesaplama kodları...
        else:
            st.warning("⚠️ Benzerlik analizi için lütfen bir 'Referans Verisi' yükleyin.")

else:
    st.info("👈 Başlamak için sol menüden bir veya birden fazla test dosyası yükleyin.")
