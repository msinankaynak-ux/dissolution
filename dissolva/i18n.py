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
    "Work with your own dissolution data, save projects and export reports.": "Kendi dissolüsyon verinizle çalışın, projeleri kaydedin ve rapor dışa aktarın.",
    "Free during beta · your dissolution data is never stored.": "Beta sürecinde ücretsiz · dissolüsyon verileriniz asla saklanmaz.",
    "— or —": "— veya —",
    "Explore the demo": "Demoyu keşfet",
    "Browse the full app with example profiles — no sign-in needed.": "Örnek profillerle tüm uygulamayı gezin — giriş gerekmez.",
    # ── Account dialog ──
    "My account": "Hesabım",
    " plan · free during beta": " plan · beta sürecinde ücretsiz",
    "All 62 models, f1/f2 and bootstrap are unlocked.": "Tüm 62 model, f1/f2 ve bootstrap kullanıma açık.",
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
    "All 62 kinetic models, f1/f2 similarity and bootstrap f2 are unlocked for you.": "Tüm 62 kinetik model, f1/f2 benzerliği ve bootstrap f2 senin için açık.",
    "What best describes your role?": "Rolünüzü en iyi ne tanımlar?",
    "(optional — helps us improve)": "(opsiyonel — geliştirmemize yardımcı olur)",
    "Get started": "Başla",
    # ── Header / session ──
    "New Session": "Yeni Oturum",
    "Load demo": "Demo yükle",
    "Try Free": "Ücretsiz Dene",
    "Start a new session?": "Yeni oturum başlatılsın mı?",
    "Demo profiles loaded — open Data Input or Kinetic Model Fitting.": "Demo profilleri yüklendi — Veri Girişi veya Kinetik Model Uyarlama'yı açın.",
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
    # ── My Records page (compliance) ──
    "My Records": "Kayıtlarım",
    "Compliance mode — your analyses are saved server-side with an immutable, time-stamped audit trail (21 CFR Part 11 readiness). Records stay private to your account.": "Uyum modu — analizleriniz, değişmez ve zaman damgalı bir denetim iziyle (21 CFR Part 11 hazırlığı) sunucuda saklanır. Kayıtlar yalnızca hesabınıza özeldir.",
    '**Compliance records are not enabled on this deployment.**\n\nSaving/loading with an audit trail needs the backend configured (`st.secrets["backend"]`). Until then, your work is kept locally in your browser (autosave) only.': '**Uyum kayıtları bu dağıtımda etkin değil.**\n\nDenetim iziyle kaydetme/yükleme için backend yapılandırılmalı (`st.secrets["backend"]`). O zamana kadar çalışmanız yalnızca tarayıcınızda (otomatik taslak) tutulur.',
    "Sign in (sidebar) to save and load your records — records are tied to your account.": "Kayıtlarınızı kaydedip yüklemek için (kenar çubuğundan) giriş yapın — kayıtlar hesabınıza bağlıdır.",
    "Could not determine your account email; please sign in again.": "Hesap e-postanız belirlenemedi; lütfen tekrar giriş yapın.",
    "Audit trail: **intact** ✅ (hash-chain verified)": "Denetim izi: **bütün** ✅ (hash-zinciri doğrulandı)",
    "Audit trail: **TAMPER DETECTED** ⚠️ — the audit log hash-chain does not verify.": "Denetim izi: **KURCALAMA TESPİT EDİLDİ** ⚠️ — denetim kaydı hash-zinciri doğrulanmıyor.",
    "Audit trail status: unavailable.": "Denetim izi durumu: kullanılamıyor.",
    "Save current work": "Çalışmayı kaydet",
    "💡 Your work is auto-kept in your browser so you never lose it. **Saving here** writes a permanent, audit-trailed record on the server — nothing is written to your records until you click Save/Update.": "💡 Çalışmanız tarayıcınızda otomatik saklanır, asla kaybolmaz. **Buradan kaydetmek**, sunucuda kalıcı ve denetim-izli bir kayıt oluşturur — Kaydet/Güncelle'ye basmadan kayıtlarınıza hiçbir şey yazılmaz.",
    "this record": "bu kayıt",
    "Load or enter dissolution profiles first, then save them here.": "Önce dissolüsyon profillerini yükleyin veya girin, sonra buraya kaydedin.",
    "📂 Editing saved record:": "📂 Düzenlenen kayıt:",
    "⬆️ Update “{name}”": "⬆️ Güncelle “{name}”",
    "Overwrite the loaded record (saves a new version; the previous version stays in the audit trail).": "Yüklü kaydın üzerine yaz (yeni bir versiyon kaydeder; önceki versiyon denetim izinde kalır).",
    "➕ Save as a new record": "➕ Yeni kayıt olarak kaydet",
    "New record name": "Yeni kayıt adı",
    "💾 Save new": "💾 Yeni kaydet",
    "Record updated — new version saved.": "Kayıt güncellendi — yeni versiyon kaydedildi.",
    "Update failed: {e}": "Güncelleme başarısız: {e}",
    "Saved as a new record.": "Yeni kayıt olarak kaydedildi.",
    "Save failed: {e}": "Kaydetme başarısız: {e}",
    "Saved records": "Kayıtlı analizler",
    "Could not load your records: {e}": "Kayıtlarınız yüklenemedi: {e}",
    "No saved records yet.": "Henüz kayıt yok.",
    "v{v} · updated {d}": "v{v} · güncellendi {d}",
    "Load": "Yükle",
    "Loaded “{name}”.": "Yüklendi “{name}”.",
    "Could not read this record's data.": "Bu kaydın verisi okunamadı.",
    "Load failed: {e}": "Yükleme başarısız: {e}",
    "Delete": "Sil",
    "Reason for deletion (kept in audit trail)": "Silme gerekçesi (denetim izinde saklanır)",
    "Confirm delete": "Silmeyi onayla",
    "Record archived (soft-deleted).": "Kayıt arşivlendi (yumuşak silme).",
    "Delete failed: {e}": "Silme başarısız: {e}",
    "Cancel": "İptal",
    "🔎 Audit trail — your activity (who · what · when)": "🔎 Denetim izi — etkinliğiniz (kim · ne · ne zaman)",
    "Audit trail unavailable: {e}": "Denetim izi kullanılamıyor: {e}",
    "No audit entries yet.": "Henüz denetim kaydı yok.",
    "When (UTC)": "Zaman (UTC)",
    "Action": "İşlem",
    "Record": "Kayıt",
    "Immutable, hash-chained log (21 CFR Part 11 §11.10(e)) — append-only, cannot be edited.": "Değişmez, hash-zincirli kayıt (21 CFR Part 11 §11.10(e)) — yalnızca ekleme, düzenlenemez.",
}
