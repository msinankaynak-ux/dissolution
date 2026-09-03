"""Feature access helpers (beta): Pro / industry features are unlocked for
admin e-mails plus the `[access] pro_emails` list in Streamlit Secrets."""
import streamlit as st
from dissolva import auth as _auth


def _emails(section, key):
    out = set()
    try:
        for x in st.secrets.get(section, {}).get(key) or []:
            out.add(str(x).strip().lower())
    except Exception:
        pass
    return out


def pro_emails():
    e = _emails("admin", "emails") | _emails("access", "pro_emails")
    if not e:
        e.add("msinankaynak@gmail.com")  # fail-safe owner
    return e


def current_email():
    try:
        return ((_auth.current_user() or {}).get("email") or "").strip().lower()
    except Exception:
        return ""


def is_pro():
    """True when the signed-in user may use Pro / industry features."""
    if not _auth.is_authenticated():
        return False
    return current_email() in pro_emails()
