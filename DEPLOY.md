# DissolvA — Yayına Alma (Streamlit Community Cloud)

Beta planı: **isteğe bağlı giriş**, app Streamlit Cloud'da, landing (dissolva.app) yeni app'e yönlenir.

## Faz 1 — App'i deploy et (SEN, panelden)
1. https://share.streamlit.io → **GitHub ile giriş** (msinankaynak-ux hesabı).
2. **Create app → Deploy a public app from GitHub** (private repo da olur; Streamlit'e repo erişimi izni iste — bir kez).
3. Ayarlar:
   - **Repository:** `msinankaynak-ux/dissolution`
   - **Branch:** `dev`
   - **Main file path:** `app.py`
   - **App URL (subdomain):** özel bir ad seç, ör. `dissolva-app` → adres `https://dissolva-app.streamlit.app`
     *(Bu adresi baştan seçmek önemli; Google girişini buna göre ayarlayacağız.)*
   - **Advanced settings → Python version: 3.11**
   - Secrets'ı ŞİMDİLİK BOŞ bırak (giriş Faz 3'te eklenecek; şimdilik açık mod).
4. **Deploy** → build bitince app açılır (giriş yok = açık mod, normal).
5. Çıkan **adresi** Claude'a ver → Faz 2 ve 3'ü o yapar/yönlendirir.

> Not: Eski `dissanalyze.streamlit.app` uygulamasına dokunmuyoruz; bu YENİ bir app.

## Faz 2 — Landing linkini güncelle (CLAUDE)
- `website/index.html` içindeki "Launch App" butonlarını eski `dissanalyze.streamlit.app`'ten
  yeni adrese çevir → `dissolva-website` repo'ya push → GitHub Pages otomatik günceller.

## Faz 3 — Google girişini canlı adrese bağla (BİRLİKTE)
1. Google Cloud Console → OAuth client (Web), **Authorized redirect URI:**
   `https://<senin-adresin>.streamlit.app/oauth2callback`
2. Streamlit Cloud → app **Settings → Secrets** → şu TOML'u yapıştır (AUTH_SETUP.md'deki gibi):
   ```toml
   [auth]
   redirect_uri = "https://<senin-adresin>.streamlit.app/oauth2callback"
   cookie_secret = "<64 hex>"
   [auth.google]
   client_id = "..."
   client_secret = "..."
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
3. App otomatik yeniden başlar → sidebar'da **"Google ile giriş yap"** çıkar. Test et.

## Sonra (opsiyonel, üretim hijyeni)
- `dev` → `main` merge edip Streamlit Cloud'u `main` branch'ine çevirmek (production = main).
- Kalıcı profil kaydı için veritabanı (Firestore) — Phase 3.
