import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- KİNETİK MODELLER (LİTERATÜR SETİ) ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

def calculate_mdt(time, means):
    delta_t = np.diff(time, prepend=0)
    mid_q = np.diff(means, prepend=0)
    mid_t = time - (delta_t / 2)
    return np.sum(mid_t * mid_q) / np.sum(mid_q) if np.sum(mid_q) > 0 else 0

st.set_page_config(page_title="PharmTech Lab Pro v2.5", layout="wide")
st.title("🔬 PharmTech Lab: Tam Kapsamlı Dissolüsyon Analizi")

def process_data(file):
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
        values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        return time, values.mean(axis=1).values, values.std(axis=1).values, values
    except:
        return None, None, None, None

# --- DOSYA YÜKLEME ---
col_u1, col_u2 = st.columns(2)
with col_u1:
    main_file = st.file_uploader("Test (Numune) Verisi Yükleyin", type=['xlsx', 'csv'], key="main_u")
with col_u2:
    ref_file = st.file_uploader("Referans Verisi Yükleyin (Opsiyonel)", type=['xlsx', 'csv'], key="ref_u")

if main_file:
    time, mean_q, std_q, raw_q = process_data(main_file)
    
    # --- MODEL ANALİZİ VE HESAPLAMALAR ---
    t_fit, q_fit = time[time > 0], mean_q[time > 0]
    kin_res = []
    models = [
        ("Zero-Order", zero_order, [1]),
        ("First-Order", first_order, [0.1]),
        ("Higuchi", higuchi, [1]),
        ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
        ("Hixson-Crowell", hixson_crowell, [0.01]),
        ("Weibull", weibull, [100, 1])
    ]
    
    best_model_func = None
    best_params = None
    
    for name, func, p0 in models:
        try:
            popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
            y_pred = func(t_fit, *popt)
            r2 = r2_score(q_fit, y_pred)
            rss = np.sum((q_fit - y_pred)**2)
            aic = calculate_aic(len(t_fit), rss, len(p0))
            
            mech = ""
            if name == "Korsmeyer-Peppas":
                n = popt[1]
                mech = "Fickian" if n <= 0.45 else "Anomalous" if n < 0.89 else "Case-II"
            
            kin_res.append({"Model": name, "R2": r2, "AIC": aic, "Params": popt, "Mekanizma": mech})
        except: continue

    # --- GÖRSELLEŞTİRME PANELİ ---
    st.subheader("📊 Çözünme Profili ve Model Uyumu")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(time, mean_q, yerr=std_q, fmt='ok', label="Deneysel (Test)", capsize=5)
    
    t_plot = np.linspace(0, time.max(), 100)
    res_df = pd.DataFrame(kin_res).sort_values("AIC")
    
    if not res_df.empty:
        best_row = res_df.iloc[0]
        m_name = best_row['Model']
        m_func = next(f for n, f, p in models if n == m_name)
        ax.plot(t_plot, m_func(t_plot, *best_row['Params']), 'r--', label=f"En İyi Fit: {m_name}")

    if ref_file:
        tr, mr, sr, _ = process_data(ref_file)
        ax.errorbar(tr, mr, yerr=sr, fmt='s', color='gray', label="Referans", alpha=0.5, capsize=3)
        
    ax.set_xlabel("Zaman (dakika)")
    ax.set_ylabel("Kümülatif Salım (%)")
    ax.legend()
    st.pyplot(fig)

    # GRAFİĞİ İNDİR BUTONU
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button("🖼️ Grafiği PNG Olarak İndir", buf.getvalue(), "dissolüsyon_profili.png", "image/png")

    # --- VERİ TABLOLARI VE RAPORLAMA ---
    tabs = st.tabs(["📈 Kinetik Sonuçlar", "📝 Karşılaştırma Raporu", "📋 Ham Veri"])
    
    with tabs[0]:
        st.write("**Model İstatistikleri (AIC Sıralı)**")
        st.dataframe(res_df[["Model", "R2", "AIC", "Mekanizma"]], use_container_width=True)
        
    with tabs[1]:
        st.subheader("Akademik Değerlendirme Raporu")
        col1, col2, col3 = st.columns(3)
        mdt_val = calculate_mdt(time, mean_q)
        col1.metric("MDT (Test)", f"{mdt_val:.2f} dk")
        
        if ref_file:
            mdt_ref = calculate_mdt(tr, mr)
            col2.metric("MDT (Referans)", f"{mdt_ref:.2f} dk")
            
            # f1, f2 hesabı
            R, T = mr, mean_q
            f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
            mse = np.mean((R - T)**2)
            f2 = 50 * np.log10((1 + mse)**-0.5 * 100)
            st.info(f"**Benzerlik Sonucu:** f1 = {f1:.2f}, f2 = {f2:.2f}")
            if f2 >= 50: st.success("✅ Profiller benzer kabul edilir (f2 ≥ 50).")
            else: st.warning("❌ Profiller benzer değildir (f2 < 50).")

    with tabs[2]:
        st.write("**Yüklenen Ham Veri Özeti**")
        st.dataframe(raw_q)
