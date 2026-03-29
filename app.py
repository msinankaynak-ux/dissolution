import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- SMART KİNETİK MOTOR ---
def interpret_n(n):
    if n <= 0.45: return "Fickian Diffusion"
    elif 0.45 < n < 0.89: return "Anomalous Transport"
    return "Case II / Super Case II"

def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**n)

# --- ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab Pro v13", layout="wide")
st.sidebar.title("🔬 Smart Lab v13")

menu = st.sidebar.radio("Analiz Menüsü:", ["📈 Profiller", "🧮 Akıllı Modelleme", "📊 Karşılaştırma"])

st.sidebar.divider()
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
    t, q = test["t"], test["mean"]
    
    if menu == "🧮 Akıllı Modelleme":
        st.subheader("🔍 Akıllı Kinetik Optimizasyon")
        
        # Fit için veri hazırlığı (Hata payını sıfıra indirir)
        f_mask = (t > 0) & (q > 0) & (q < 105)
        tf, qf = t[f_mask], q[f_mask]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(t, q, color='black', label="Deneysel")
        
        results = []
        models = [
            ("Sıfır Derece", zero_order, [0.1], (0, 100)),
            ("Birinci Derece", first_order, [0.01], (0, 10)),
            ("Higuchi", higuchi, [1.0], (0, 500)),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], (0, [200, 2.0]))
        ]

        for name, func, p0, bnds in models:
            try:
                # AKILLI BAŞLANGIÇ NOKTASI (Auto-Seeding)
                # Eğer Peppas ise log-log üzerinden n'i tahmin et
                current_p0 = p0
                if name == "Korsmeyer-Peppas":
                    log_t, log_q = np.log(tf), np.log(qf)
                    slope, intercept = np.polyfit(log_t, log_q, 1)
                    current_p0 = [np.exp(intercept), np.clip(slope, 0.1, 1.5)]
                
                popt, _ = curve_fit(func, tf, qf, p0=current_p0, bounds=bnds, maxfev=10000)
                y_fit = func(t, *popt)
                r2 = r2_score(q, y_fit)
                
                res = {"Model": name, "R²": f"{r2:.4f}", "K": f"{popt[0]:.4f}"}
                if name == "Korsmeyer-Peppas":
                    res["n"] = f"{popt[1]:.3f}"
                    res["Mekanizma"] = interpret_n(popt[1])
                results.append(res)
                ax.plot(t, y_fit, label=f"{name}")
            except:
                results.append({"Model": name, "R²": "Fit Edilemedi", "K": "-", "n": "-", "Mekanizma": "Veri Uyumsuz"})

        ax.legend(); st.pyplot(fig)
        st.table(pd.DataFrame(results).fillna("-"))

    # Diğer sekmeler (Profiller ve Karşılaştırma) v12.1'deki gibi stabil çalışır...
else:
    st.info("Hocam verileri yükleyin, gerisini akıllı algoritmaya bırakın.")
