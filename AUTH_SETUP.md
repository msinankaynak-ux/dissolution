# DissolvA — Google Giriş (Authentication) Kurulumu

DissolvA, Streamlit'in **yerleşik OIDC** girişini kullanır (`st.login` / `st.user`).
Yapılandırma yoksa uygulama **açık modda** çalışır (giriş zorunlu değil) — yani
kurulum yapmadan da her şey çalışmaya devam eder. Gerçek Google girişini açmak için:

## 1. Google OAuth client oluştur (tek seferlik)
1. https://console.cloud.google.com/ → bir proje seç veya oluştur.
2. **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
3. Application type: **Web application**.
4. **Authorized redirect URIs** kısmına, uygulamayı çalıştırdığın adres + `/oauth2callback` ekle:
   - Varsayılan: `http://localhost:8501/oauth2callback`
   - Özel port kullanıyorsan (ör. 8765): `http://localhost:8765/oauth2callback`
5. Oluştur → **Client ID** ve **Client Secret** değerlerini kopyala.

## 2. secrets.toml dosyasını doldur
```bash
cd ~/dissolva/app
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# cookie_secret üret:
python -c "import secrets; print(secrets.token_hex(32))"
```
`.streamlit/secrets.toml` içine:
- `redirect_uri` → Google'a girdiğin URI ile **birebir aynı** olmalı (port dahil).
- `cookie_secret` → yukarıda ürettiğin hex değer.
- `[auth.google]` altına `client_id` ve `client_secret`.

> ⚠️ `secrets.toml` gerçek anahtar içerir → `.gitignore`'da hariç tutuldu, **git'e girmez**.

## 3. Çalıştır
```bash
streamlit run app.py
```
Sol menüde **"🔑 Google ile giriş yap"** butonu çıkar. Giriş yapınca kullanıcı
kartı (ad + e-posta) görünür, **"Çıkış yap"** ile çıkılır.

## Notlar
- **Tier (üyelik) enforcement henüz KAPALI** — giriş yapılsa da tüm özellikler açık.
  Ücretli kapılar (`auth.require_login`, `state.require_tier`) ileride bağlanacak (Phase 6, Stripe).
- Kod: `dissolva/auth.py` (giriş sarmalayıcı), `app.py` sidebar'da `auth.render_sidebar_auth()`.
- GitHub girişi de eklenebilir: `[auth.github]` sağlayıcısı ekleyip butona ikinci `st.login("github")` bağla.
