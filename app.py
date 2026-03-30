# --- 1. LİTERATÜR BİLGİ BANKASI (YORUM İÇİN) ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Bu model, zamandan bağımsız sabit hızda salımı açıklar; genellikle kontrollü salım sistemleri için idealdir.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır; konvansiyonel dozaj formları için karakteristiktir.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Bu model, matris sistemlerinden difüzyon temelli salımı açıklar; zamanın karekökü ile orantılı bir salım gözlenir.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Salım mekanizması 'n' üsteli ile tanımlanır; hem difüzyon hem de polimer şişmesini bir arada açıklar.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Bu model, 'Erozyon ve Şişme Temelli' bir yaklaşım olup, ilaç parçacıklarının yüzey alanı ve çapının zamanla küçüldüğü durumları açıklar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Bu model, salımdaki difüzyonel katkı ile polimer zincir relaksasyonu (erozyon) katkısını matematiksel olarak birbirinden ayırır.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Bu ampirik model, profilin şeklini (sigmoid vb.) ve gecikme süresini (Td) karakterize eder."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics. This model describes constant release rate independent of time, ideal for controlled release systems.",
        "Birinci Derece": "fits First-Order kinetics. The release rate is concentration-dependent, typical for conventional dosage forms.",
        "Higuchi": "fits the Higuchi model, describing diffusion-based release from matrix systems proportional to the square root of time.",
        "Korsmeyer-Peppas": "fits the Korsmeyer-Peppas model. The mechanism is defined by the 'n' exponent, explaining both diffusion and polymer swelling.",
        "Hixson-Crowell": "fits Hixson-Crowell kinetics. This is an erosion/swelling based model explaining cases where surface area and particle diameter decrease over time.",
        "Peppas-Sahlin": "fits the Peppas-Sahlin model, mathematically separating the contributions of diffusion and polymer relaxation (erosion).",
        "Weibull (w/ Td)": "fits the Weibull model, characterizing the profile shape and dissolution lag time (Td)."
    }
}

# --- 2. OTOMATİK RAPOR OLUŞTURUCU (KODUN İÇİNE EKLENECEK KISIM) ---
def generate_summary_paragraph(best_model_name, lang):
    knowledge = MODEL_KNOWLEDGE[lang]
    if lang == "Türkçe":
        base_text = f"**📊 Otomatik Analiz Sonucu:** Test preparatımız **{best_model_name}** {knowledge.get(best_model_name, '')}"
    else:
        base_text = f"**📊 Automated Analysis Result:** The test preparation **{knowledge.get(best_model_name, '')}**"
    return base_text

# --- 3. UYGULAMA İÇİNDE GÖSTERİM ---
# (df_res tablosunun hemen altına eklenecek)
if "2." in menu:
    # ... (Kinetik hesaplamalar yapıldıktan sonra)
    best_model = df_res.loc[best_idx, "Model"]
    
    st.divider()
    st.subheader("📝 Akademik Değerlendirme / Academic Evaluation")
    
    # Paragrafı oluştur ve ekrana bas
    report_para = generate_summary_paragraph(best_model, selected_lang)
    st.info(report_para)
    
    # Veri analizi sonuçlarını metne dökme (Örn: R2 ve AIC değerleri ile)
    if selected_lang == "Türkçe":
        st.write(f"Model uyumu **R²: {df_res.loc[best_idx, 'R²']:.4f}** ve **AIC: {df_res.loc[best_idx, 'AIC']:.2f}** değerleri ile doğrulanmıştır.")
    else:
        st.write(f"Model suitability is verified with **R²: {df_res.loc[best_idx, 'R²']:.4f}** and **AIC: {df_res.loc[best_idx, 'AIC']:.2f}** values.")
