import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- GÜVENLİ MATEMATİKSEL FONKSİYONLAR ---
def safe_div(n, d): return n / d if d != 0 else 0

def interpret_n(n):
    if n <= 0.45: return "Fickian Diffusion (Case I)"
    elif 0.45 < n < 0.89: return "Anomalous Transport (Non-Fickian)"
    elif n >= 0.89: return "Case II Transport (Relaxation/Erosion)"
    return "Unknown"

# --- MODELLER ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**n)

# --- UYGULAMA ---
st.set_page_config(page_title="PharmTech Lab Pro v10.0", layout="wide")
st.sidebar.title("🚀 Pro Engineer v10.0")

test_file = st.sidebar.file_uploader("Test Verisi (Excel/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_data(file):
    if not file: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        # Sadece NaN temizliği yapıyoruz, modelleri kısıtlamıyoruz
        mask = ~np.isnan(t) & (t >= 0)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except: return None

test = load_data(test_file)
ref = load_data(ref_file)

if test:
    t, q = test["t"], test["mean"]
    
    # --- MODEL BAĞIMSIZ ANALİZ ---
    st.subheader("📋 Karşılaştırmalı Parametreler")
    c1, c2, c3 = st.columns(3)
    
    # MDT ve DE hesaplama
    dq = np.maximum(np.diff(q, prepend=0), 0)
    mid_t = [t[0]/2] + [(t[i] + t[i-1])/2 for i in range(1, len(t))]
    mdt = safe_div(np.sum(dq * np.array(mid_t)), np.max(q))
    de = safe_div(np.trapz(q, t) * 100, np.max(t) * 100)
    
    c1.metric("DE (Dissolution Efficiency)", f"%{de:.2f}")
    c2.metric("MDT (Mean Dissolution Time)", f"{mdt:.2f} dk")
    
    if ref:
        sum_sq = np.sum((ref["mean"] - q)**2)
        f2 = 50 * np.log10(100 / np.sqrt(1 + sum_sq/len(q)))
        c3.metric("f2 (Similarity Factor)", f"{f2:.2f}")

    st.divider()

    # --- KİNETİK MODELLER VE FİT ---
    st.subheader("🧮 Kinetik Modelleme ve Salım Mekanizması")
    
    # Fit için 0 zamanını ve 0 salımını filtrele (logaritmik modeller için şart)
    fit_mask = (t > 0) & (q > 0)
    tf, qf = t[fit_mask], q[fit_mask]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(t, q, color='black', label="Deneysel Veri", zorder=3)
    
    results = []
    # Modeller: (İsim, Fonksiyon, [Başlangıç Parametreleri], [Alt Sınır], [Üst Sınır])
    model_list = [
        ("Sıfır Derece", zero_order, [0.1], [0], [100]),
        ("Birinci Derece", first_order, [0.01], [0], [5]),
        ("Higuchi", higuchi, [1.0], [0], [500]),
        ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.01], [100, 2.0])
    ]

    for name, func, p0, b_low, b_up in model_list:
        try:
            popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(b_low, b_up), maxfev=10000)
            y_fit = func(tf, *popt)
            r2 = r2_score(qf, y_fit)
            
            # Mekanizma Yorumu (Sadece Peppas için)
            note = ""
            if name == "Korsmeyer-Peppas":
                note = interpret_n(popt[1])
            
            results.append({
                "Model": name, 
                "R²": f"{r2:.4f}", 
                "K Değeri": f"{popt[0]:.4f}", 
                "n Değeri": f"{popt[1]:.4f}" if len(popt)>1 else "-",
                "Mekanizma / Not": note
            })
            ax.plot(tf, y_fit, label=f"{name} (R²:{r2:.3f})")
        except:
            results.append({"Model": name, "R²": "Hata", "K Değeri": "-", "n Değeri": "-", "Mekanizma / Not": "Fit Edilemedi"})

    ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Kümülatif Salım (%)"); ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.table(pd.DataFrame(results))

else:
    st.info("Hocam verileri yüklediğinde tüm kinetikler burada görünecek.")
