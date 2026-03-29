import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- HESAPLAMA FONKSİYONLARI ---
def calculate_de(t, q):
    if len(t) < 2: return 0
    auc = np.trapz(q, t)
    total_area = np.max(t) * 100
    return (auc / total_area) * 100 if total_area > 0 else 0

def calculate_mdt(t, q):
    if len(t) < 2 or np.max(q) <= 0: return 0
    delta_q = np.diff(q, prepend=0)
    mid_t = []
    for i in range(len(t)):
        if i == 0: mid_t.append(t[i]/2)
        else: mid_t.append((t[i] + t[i-1])/2)
    mdt = np.sum(delta_q * np.array(mid_t)) / np.max(q)
    return mdt

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PharmTech Lab Pro v7.1", layout="wide")

st.sidebar.title("🔬 PharmTech Pro")
menu = st.sidebar.radio("Analiz Paneli:", ["📈 Profil Analizi", "📊 Model-Bağımsız Özet", "🧮 Kinetik Modelleme"])

test_file = st.sidebar.file_uploader("Test Verisi", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except: return None

test = load_data(test_file)
ref = load_data(ref_file)

if test:
    t_t, m_t, s_t = test["t"], test["mean"], test["std"]

    if menu == "📊 Model-Bağımsız Özet":
        st.subheader("📋 Farmasötik Karşılaştırma Özeti")
        
        # Test Verisi Hesaplamaları
        de_t = calculate_de(t_t, m_t)
        mdt_t = calculate_mdt(t_t, m_t)
        mdr_t = m_t.max() / mdt_t if mdt_t > 0 else 0
        
        # UI Kartları
        c1, c2, c3 = st.columns(3)
        c1.metric("DE (Çözünme Verimi)", f"%{de_t:.2f}")
        c2.metric("MDT (Ort. Çözünme Süresi)", f"{mdt_t:.2f} dk")
        c3.metric("MDR (Ort. Çözünme Hızı)", f"{mdr_t:.2f} %/dk")

        st.divider()

        if ref:
            t_r, m_r = ref["t"], ref["mean"]
            if len(m_r) == len(m_t):
                # f1 / f2 Hesaplama
                f1 = (np.sum(np.abs(m_r - m_t)) / np.sum(m_r)) * 100
                diff_sq = (m_r - m_t)**2
                f2 = 50 * np.log10((1 + (1/len(m_r)) * np.sum(diff_sq))**-0.5 * 100)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("### Benzerlik Analizi")
                    st.success(f"**f2 (Similarity):** {f2:.2f}") if f2 >= 50 else st.error(f"**f2 (Similarity):** {f2:.2f}")
                    st.info(f"**f1 (Difference):** {f1:.2f}")
                
                with col_b:
                    st.write("### Referans Verileri")
                    de_r = calculate_de(t_r, m_r)
                    st.write(f"Referans DE: %{de_r:.2f}")
                    st.write(f"Referans MDT: {calculate_mdt(t_r, m_r):.2f} dk")
            else:
                st.warning("⚠️ Test ve Referans zaman noktaları (satır sayısı) uyuşmuyor!")

        # Akademik Bilgi Notu
        with st.expander("📝 Parametrelerin Anlamı ve FDA Kriterleri"):
            st.markdown("""
            - **f2 Faktörü:** İki profilin benzer kabul edilmesi için 50-100 arasında olmalıdır.
            - **f1 Faktörü:** İki profil arasındaki farkı gösterir, 0-15 arası kabul edilebilirdir.
            - **MDT (Mean Dissolution Time):** İlacın salım hızı karakteristiğidir. Düşük değer hızlı salımı temsil eder.
            - **DE (Dissolution Efficiency):** Eğri altındaki alanın toplam dikdörtgen alana oranıdır.
            """)

    elif menu == "📈 Profil Analizi":
        st.subheader("📈 Kümülatif Çözünme Grafiği")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_t, m_t, yerr=s_t, fmt='-ob', label="Test", capsize=5)
        if ref: ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.6)
        ax.set_ylim(0, 105); ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

else:
    st.info("👋 Hocam, lütfen sol panelden verilerinizi yükleyin.")
