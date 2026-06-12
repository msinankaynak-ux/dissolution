"""DissolvA i18n — lightweight EN/TR localisation.

t(s) returns the Turkish string for `s` when the active language is 'tr' and a
translation exists; otherwise it returns `s` unchanged (English is the source of
truth). Technical terms (kinetic-model names, f1/f2, R²/AIC, USP apparatus, API)
deliberately stay English — standard usage in Turkish pharma practice.

Language lives in st.session_state['lang'] ('en' | 'tr'), defaults to 'en', and
is persisted to the user profile (alongside theme/role) once the backend stores
it. tt(key, **kw) formats a translated template (for strings with placeholders).
"""
import streamlit as st

LANGS = {"English": "en", "Türkçe": "tr"}


def get_lang() -> str:
    return st.session_state.get("lang", "en")


def t(s: str) -> str:
    if get_lang() == "tr":
        return _TR.get(s, s)
    return s


def tt(s: str, **kw) -> str:
    return t(s).format(**kw)


# English -> Turkish. Phase 1 = UI shell (nav, gate, account, welcome, header,
# privacy, roles, Academy chrome). Page bodies + content dicts: later phases.
_TR = {
    # ── Navigation: category titles ──
    "Configuration & Setup": "Yapılandırma ve Kurulum",
    "Predictive Analysis": "Öngörücü Analiz",
    "Results & Documentation": "Sonuçlar ve Belgeler",
    # ── Navigation: page labels (routing values stay English) ──
    "Method Settings": "Yöntem Ayarları",
    "Analytical Settings": "Analitik Ayarlar",
    "Data Input": "Veri Girişi",
    "Template Builder": "Şablon Oluşturucu",
    "Kinetic Model Fitting": "Kinetik Model Uyarlama",
    "Statistical Analysis": "İstatistiksel Analiz",
    "f1 & f2 Similarity": "f1 & f2 Benzerlik",
    "Bootstrap f2": "Bootstrap f2",
    "IVIVC Correlation": "IVIVC Korelasyonu",
    "Excel Reporting": "Excel Raporlama",
    "References": "Kaynaklar",
    "API Information": "API (Etken Madde) Bilgisi",
    # ── Gate ──
    "Sign in to start your analysis": "Analizinize başlamak için giriş yapın",
    "Work with your own dissolution data, save projects and export reports.":
        "Kendi dissolüsyon verinizle çalışın, projeleri kaydedin ve rapor dışa aktarın.",
    "Free during beta · your dissolution data is never stored.":
        "Beta sürecinde ücretsiz · dissolüsyon verileriniz asla saklanmaz.",
    "— or —": "— veya —",
    "Explore the demo": "Demoyu keşfet",
    "Browse the full app with example profiles — no sign-in needed.":
        "Örnek profillerle tüm uygulamayı gezin — giriş gerekmez.",
    # ── Account dialog ──
    "My account": "Hesabım",
    " plan · free during beta": " plan · beta sürecinde ücretsiz",
    "All 62 models, f1/f2 and bootstrap are unlocked.":
        "Tüm 62 model, f1/f2 ve bootstrap kullanıma açık.",
    "Appearance": "Görünüm",
    "Your role": "Rolünüz",
    "(helps us improve — optional)": "(geliştirmemize yardımcı olur — opsiyonel)",
    "Select your role…": "Rolünüzü seçin…",
    "Save": "Kaydet",
    "Preferences saved.": "Tercihler kaydedildi.",
    "Language": "Dil",
    "Account": "Hesap",
    "Account & settings": "Hesap ve ayarlar",
    "Log out": "Çıkış yap",
    # ── Welcome ──
    "Welcome to DissolvA": "DissolvA'ya Hoş Geldiniz",
    "Welcome, {name}! You're all set.": "Hoş geldin, {name}! Her şey hazır.",
    " · free during beta": " · beta sürecinde ücretsiz",
    "All 62 kinetic models, f1/f2 similarity and bootstrap f2 are unlocked for you.":
        "Tüm 62 kinetik model, f1/f2 benzerliği ve bootstrap f2 senin için açık.",
    "What best describes your role?": "Rolünüzü en iyi ne tanımlar?",
    "(optional — helps us improve)": "(opsiyonel — geliştirmemize yardımcı olur)",
    "Get started": "Başla",
    # ── Header / session ──
    "New Session": "Yeni Oturum",
    "Load demo": "Demo yükle",
    "Try Free": "Ücretsiz Dene",
    "Start a new session?": "Yeni oturum başlatılsın mı?",
    "Demo profiles loaded — open Data Input or Kinetic Model Fitting.":
        "Demo profilleri yüklendi — Veri Girişi veya Kinetik Model Uyarlama'yı açın.",
    # ── Privacy dialog ──
    "🔒 Data privacy": "🔒 Veri gizliliği",
    # ── Roles ──
    "Formulation Development": "Formülasyon Geliştirme",
    "Analytical Development / R&D": "Analitik Geliştirme / Ar-Ge",
    "Quality Control (QC) / QA": "Kalite Kontrol (QC) / QA",
    "Regulatory Affairs / CMC": "Regülasyon İşleri / CMC",
    "Biopharmaceutics / Bioequivalence": "Biyofarmasötik / Biyoeşdeğerlik",
    "Process / Manufacturing Sciences": "Proses / Üretim Bilimleri",
    "Academia / Researcher": "Akademi / Araştırmacı",
    "Student": "Öğrenci",
    "Other": "Diğer",
    "Prefer not to say": "Belirtmek istemiyorum",
    # ── Academy chrome ──
    "Learn": "Öğren",
    "DissolvA Academy": "DissolvA Akademi",
    "**Welcome, {name}!** You're all set.": "**Hoş geldin, {name}!** Her şey hazır.",
    "🎓 DissolvA Academy": "🎓 DissolvA Akademi",
    "📐 Kinetic Models": "📐 Kinetik Modeller",
    "🧫 Dissolution Methods": "🧫 Dissolüsyon Yöntemleri",
    "Category": "Kategori",
    "Model": "Model",
    "Method": "Yöntem",
}
