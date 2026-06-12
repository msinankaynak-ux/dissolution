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
# Official Google "G" 4-color logo (data URI) for the standard Sign-in button look.
GOOGLE_G_ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4Ij48cGF0aCBmaWxsPSIjRUE0MzM1IiBkPSJNMjQgOS41YzMuNTQgMCA2LjcxIDEuMjIgOS4yMSAzLjZsNi44NS02Ljg1QzM1LjkgMi4zOCAzMC40NyAwIDI0IDAgMTQuNjIgMCA2LjUxIDUuMzggMi41NiAxMy4yMmw3Ljk4IDYuMTlDMTIuNDMgMTMuNzIgMTcuNzQgOS41IDI0IDkuNXoiLz48cGF0aCBmaWxsPSIjNDI4NUY0IiBkPSJNNDYuOTggMjQuNTVjMC0xLjU3LS4xNS0zLjA5LS4zOC00LjU1SDI0djkuMDJoMTIuOTRjLS41OCAyLjk2LTIuMjYgNS40OC00Ljc4IDcuMThsNy43MyA2YzQuNTEtNC4xOCA3LjA5LTEwLjM2IDcuMDktMTcuNjV6Ii8+PHBhdGggZmlsbD0iI0ZCQkMwNSIgZD0iTTEwLjUzIDI4LjU5Yy0uNDgtMS40NS0uNzYtMi45OS0uNzYtNC41OXMuMjctMy4xNC43Ni00LjU5bC03Ljk4LTYuMTlDLjkyIDE2LjQ2IDAgMjAuMTIgMCAyNGMwIDMuODguOTIgNy41NCAyLjU2IDEwLjc4bDcuOTctNi4xOXoiLz48cGF0aCBmaWxsPSIjMzRBODUzIiBkPSJNMjQgNDhjNi40OCAwIDExLjkzLTIuMTMgMTUuODktNS44MWwtNy43My02Yy0yLjE1IDEuNDUtNC45MiAyLjMtOC4xNiAyLjMtNi4yNiAwLTExLjU3LTQuMjItMTMuNDctOS45MWwtNy45OCA2LjE5QzYuNTEgNDIuNjIgMTQuNjIgNDggMjQgNDh6Ii8+PC9zdmc+"


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


def _initials(s: str) -> str:
    """Two-letter initials for the avatar fallback (from a name or email)."""
    s = (s or "").strip()
    if not s:
        return "U"
    if "@" in s and " " not in s:          # an email → use the local part
        s = s.split("@")[0]
    parts = [p for p in s.replace(".", " ").replace("_", " ").split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return (parts[0][:2].upper() if parts else "U")


def render_account():
    """Sidebar bottom: signed-in account row (avatar + name + email + logout)."""
    if not is_authenticated():
        return
    u = current_user()
    pic = u.get("picture") or ""
    name = u.get("name") or "User"
    email = u.get("email") or ""
    initials = _initials(name if name != "User" else email)
    if pic:
        avstyle = "background-image:url(" + pic + ");background-size:cover;background-position:center;"
        avtxt = ""
    else:
        avstyle = "background:#003171;"
        avtxt = '<span style="color:#FFCC00;font-size:0.78rem;font-weight:700;">' + initials + '</span>'
    css = (
        "<style>"
        ".st-key-acctrow{position:relative;border-top:1px solid rgba(255,255,255,0.08);margin-top:14px;padding-top:12px;}"
        ".acctrow-in{display:flex;align-items:center;gap:10px;padding-right:34px;}"
        ".acctrow-av{width:36px;height:36px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;border:2px solid rgba(255,204,0,0.45);}"
        ".acctrow-name{color:#e8edf6 !important;font-size:0.8rem;font-weight:600;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
        ".acctrow-mail{color:#9fb0d0 !important;font-size:0.66rem;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
        ".st-key-acct_logout{position:absolute !important;top:10px !important;right:0 !important;width:auto !important;flex:0 0 auto !important;}"
        ".st-key-acct_logout button{background:transparent !important;border:none !important;min-height:0 !important;padding:6px !important;box-shadow:none !important;}"
        ".st-key-acct_logout button p{display:none !important;}"
        ".st-key-acct_logout button *{color:#9fb0d0 !important;}"
        ".st-key-acct_logout button:hover *{color:#FFCC00 !important;}"
        "</style>"
    )
    html = (
        css
        + '<div class="acctrow-in"><div class="acctrow-av" style="' + avstyle + '">' + avtxt + '</div>'
        + '<div style="min-width:0;"><div class="acctrow-name">' + name + '</div>'
        + '<div class="acctrow-mail">' + email + '</div></div></div>'
    )
    with st.container(key="acctrow"):
        st.markdown(html, unsafe_allow_html=True)
        if st.button("Log out", icon=":material/logout:", key="acct_logout", help="Sign out"):
            _logout()


def _detect_country() -> str:
    """User's country from the BROWSER timezone (reliable, location-based).
    Avoids server-IP geolocation, which on Streamlit Cloud always returns the
    US datacenter. Falls back to '' (then the backend may geolocate)."""
    try:
        tz = getattr(getattr(st, "context", None), "timezone", None)
        if not tz:
            return ""
        import pytz
        for code, zones in pytz.country_timezones.items():
            if tz in zones:
                if code == "TR":
                    return "Türkiye"
                return (pytz.country_names.get(code, "") or "")[:80]
    except Exception:
        pass
    return ""


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
        return  # signed-in: account row rendered at sidebar bottom (render_account)

    oauth2 = OAuth2Component(cid, csec, _AUTHORIZE, _TOKEN, _TOKEN, _REVOKE)
    result = oauth2.authorize_button(
        name="Sign in with Google",
        icon=GOOGLE_G_ICON,
        redirect_uri=redirect,
        scope="openid email profile",
        key="google_login",
        extras_params={"prompt": "select_account"},
        use_container_width=False,
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
            engine_client.upsert_member(info.get("email"), info.get("name"), _detect_country())
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
