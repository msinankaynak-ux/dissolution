"""DissolvA authentication / membership module.

Streamlit yerleşik OIDC girişini (st.login / st.logout / st.user) sarmalar.
Giriş henüz yapılandırılmadıysa (secrets.toml yoksa) uygulama açık modda çalışır.
Tier (core/research/pro) enforcement burada DEĞİL — şimdilik sadece kimlik.
Gerçek tier ataması ileride Firestore/Stripe ile bağlanacak (Phase 6)."""
import streamlit as st


def auth_configured() -> bool:
    """secrets.toml içinde [auth] + bir Google sağlayıcısı tanımlı mı?"""
    try:
        return ("auth" in st.secrets) and ("google" in st.secrets["auth"])
    except Exception:
        return False


def is_authenticated() -> bool:
    """Kullanıcı giriş yapmış mı?"""
    try:
        return bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        return False


def current_user() -> dict:
    """Giriş yapan kullanıcının bilgileri (yoksa None'lar)."""
    if is_authenticated():
        return {
            "email":   getattr(st.user, "email", None),
            "name":    getattr(st.user, "name", None),
            "picture": getattr(st.user, "picture", None),
        }
    return {"email": None, "name": None, "picture": None}


def sync_session():
    """st.user → st.session_state.user_email senkronu (durum tutarlılığı)."""
    try:
        st.session_state.user_email = current_user()["email"]
    except Exception:
        pass


def render_sidebar_auth():
    """Sidebar giriş/çıkış arayüzü. Yapılandırma yoksa nazik bir not gösterir."""
    if not auth_configured():
        st.markdown(
            '<div style="padding:8px 12px;font-size:0.7rem;color:#7a8aa0;">'
            '🔓 Açık mod — giriş yapılandırılmamış'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if is_authenticated():
        u = current_user()
        _pic = (f'<img src="{u["picture"]}" style="width:30px;height:30px;border-radius:50%;">'
                if u.get("picture") else '<span style="font-size:22px;">👤</span>')
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'background:rgba(39,174,96,0.12);border:1px solid rgba(39,174,96,0.3);'
            f'border-radius:8px;padding:8px 12px;margin:6px 0;">'
            f'{_pic}'
            f'<div style="line-height:1.2;overflow:hidden;">'
            f'<div style="font-size:12px;font-weight:600;color:white;'
            f'white-space:nowrap;text-overflow:ellipsis;overflow:hidden;">{u.get("name") or "Kullanıcı"}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,0.55);'
            f'white-space:nowrap;text-overflow:ellipsis;overflow:hidden;">{u.get("email") or ""}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Çıkış yap", use_container_width=True, key="_logout_btn"):
            st.logout()
    else:
        if st.button("🔑 Google ile giriş yap", use_container_width=True, key="_login_btn"):
            st.login("google")


def require_login(feature: str = "Bu bölüm"):
    """İleride kullanılacak: bir özelliği giriş şartına bağlamak için.
    Şu an HİÇBİR YERDE çağrılmıyor (tier enforcement kapalı kararı).
    Giriş yapılmamışsa uyarı gösterir ve False döner."""
    if not auth_configured():
        return True  # açık mod
    if is_authenticated():
        return True
    st.warning(f"🔒 {feature} için giriş gerekiyor. Sol menüden Google ile giriş yapın.")
    return False
