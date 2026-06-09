"""DissolvA authentication.

Streamlit's NATIVE st.login (OIDC) is unreliable on Streamlit Community Cloud
("Missing provider for OAuth callback" — the OAuth state is kept in memory and is
lost on the callback). So we use the `streamlit-oauth` component instead, which
runs the OAuth2 + PKCE flow in a popup and returns the token directly.

Reuses the EXISTING secret `[auth.google]` (client_id/client_secret) and derives
the app-root redirect URI from `[auth].redirect_uri` (strips /oauth2callback).
When not configured (or the package is missing) the app runs in OPEN mode.
Tier (core/research/pro) enforcement is NOT here yet — identity only.
"""
import base64
import json
import streamlit as st

try:
    from streamlit_oauth import OAuth2Component
    _OAUTH_OK = True
except Exception:
    _OAUTH_OK = False

_AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN = "https://oauth2.googleapis.com/token"
_REVOKE = "https://oauth2.googleapis.com/revoke"


def _google_cfg():
    """(client_id, client_secret, app_root_redirect) or (None, None, None)."""
    try:
        a = st.secrets["auth"]
        g = a["google"]
        cid = g.get("client_id")
        csec = g.get("client_secret")
        redirect = (a.get("redirect_uri") or "").replace("/oauth2callback", "").rstrip("/")
        if cid and csec and redirect:
            return cid, csec, redirect
    except Exception:
        pass
    return None, None, None


def auth_configured() -> bool:
    cid, _, _ = _google_cfg()
    return bool(cid) and _OAUTH_OK


def is_authenticated() -> bool:
    return bool(st.session_state.get("user_email"))


def current_user() -> dict:
    return {
        "email":   st.session_state.get("user_email"),
        "name":    st.session_state.get("user_name"),
        "picture": st.session_state.get("user_picture"),
    }


def sync_session():
    """Kept for compatibility; user_email is written on login."""
    return


def _decode_id_token(token: dict) -> dict:
    """Decode the (TLS-delivered) Google id_token JWT payload for display."""
    try:
        payload = token.get("id_token", "").split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _logout():
    for k in ("user_email", "user_name", "user_picture", "auth_token"):
        st.session_state.pop(k, None)
    st.rerun()


def render_sidebar_auth():
    """Sidebar sign-in / user card. Open-mode note when not configured."""
    cid, csec, redirect = _google_cfg()
    if not (cid and _OAUTH_OK):
        st.markdown(
            '<div style="padding:8px 12px;font-size:0.7rem;color:#7a8aa0;">'
            '🔓 Open mode — sign-in not configured</div>',
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
            f'white-space:nowrap;text-overflow:ellipsis;overflow:hidden;">{u.get("name") or "User"}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,0.55);'
            f'white-space:nowrap;text-overflow:ellipsis;overflow:hidden;">{u.get("email") or ""}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Log out", use_container_width=True, key="_logout_btn"):
            _logout()
        return

    oauth2 = OAuth2Component(cid, csec, _AUTHORIZE, _TOKEN, _TOKEN, _REVOKE)
    result = oauth2.authorize_button(
        name="Sign in with Google",
        redirect_uri=redirect,
        scope="openid email profile",
        key="google_login",
        extras_params={"prompt": "select_account"},
        use_container_width=True,
        pkce="S256",
    )
    if result and "token" in result:
        info = _decode_id_token(result["token"])
        st.session_state["user_email"]   = info.get("email")
        st.session_state["user_name"]    = info.get("name")
        st.session_state["user_picture"] = info.get("picture")
        st.session_state["auth_token"]   = result["token"]
        # Register the free "core" member (best-effort; backend no-op if DB disabled).
        try:
            from dissolva import engine_client
            engine_client.upsert_member(info.get("email"), info.get("name"))
        except Exception:
            pass
        st.rerun()


def require_login(feature: str = "This feature"):
    """Gate a feature behind sign-in (not wired anywhere yet)."""
    if not auth_configured():
        return True
    if is_authenticated():
        return True
    st.warning(f"🔒 {feature} requires sign-in. Use the sidebar to sign in with Google.")
    return False
