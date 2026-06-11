"""DissolvA session state and membership (tier) helpers.
Extracted from app.py (Phase 3a modularization)."""
import streamlit as st
from dissolva import tiers as _tiers

_SS_DEFAULTS = {
    "profiles":           {},
    "fit_results":        {},
    "selected_ref_id":    None,
    "selected_test_id":   None,
    "bootstrap_results":  None,
    "project_metadata": {
        "name":        "Untitled Project",
        "description": "",
        "created":     "",
        "analyst":     "",
    },
    "active_substance": {
        "name":            "",
        "pubchem":         None,
        "bcs_class":       None,
        "fda_methods":     [],
        "selected_method": None,
        "fetch_done":      False,
    },
    # ── Membership / entitlement (filled with real auth + Stripe in Phase 2) ──
    "user_email": None,
    "tier":       "free",   # "free" | "pro" | "enterprise" (legacy core/research/pro auto-mapped)
}

def init_session_state():
    """Set default session_state keys once."""
    for _k, _v in _SS_DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v


# ── Membership tier helpers (Phase 1 skeleton; enforcement activates in Phase 2) ──
TIER_RANK = _tiers.TIER_RANK  # {"free":0,"pro":1,"enterprise":2}

def current_tier() -> str:
    """Current user's canonical membership tier (free|pro|enterprise).
    Legacy values (core/research/pro) are auto-normalized."""
    return _tiers.normalize_tier(st.session_state.get("tier", "free"))

def require_tier(min_tier: str, feature: str = "This feature") -> bool:
    """True if the feature is available to the user. During the beta everything
    is unlocked (BETA); after launch, gates on tier rank."""
    if _tiers.BETA:
        return True
    mt = _tiers.normalize_tier(min_tier)
    if TIER_RANK.get(current_tier(), 0) >= TIER_RANK.get(mt, 99):
        return True
    _upgrade_cta(feature, mt)
    return False

def _upgrade_cta(feature: str, min_tier: str):
    """Upgrade CTA for a locked feature (post-beta)."""
    t = _tiers.normalize_tier(min_tier)
    label = _tiers.TIERS.get(t, {}).get("label", t.title())
    st.warning(f"🔒 **{feature}** is a **{label}** feature.")

def _safe_profile_names():
    """Return current profile names."""
    return list(st.session_state.profiles.keys())

def _get_index(lst, val, default=0):
    """Safely find index of value in list."""
    try:
        return lst.index(val) if val in lst else default
    except Exception:
        return default

def _rename_profile(old_name: str, new_name: str):
    """Rename profile and sync all session_state."""
    if old_name == new_name or new_name.strip() == "" or old_name not in st.session_state.profiles:
        return False
    if new_name in st.session_state.profiles:
        return False  # conflict
    # Update profiles dict (preserving order)
    new_profiles = {}
    for k, v in st.session_state.profiles.items():
        new_profiles[new_name if k == old_name else k] = v
    st.session_state.profiles = new_profiles
    # Update fit_results
    if old_name in st.session_state.fit_results:
        st.session_state.fit_results[new_name] = st.session_state.fit_results.pop(old_name)
    # Update sticky selections
    if st.session_state.selected_ref_id == old_name:
        st.session_state.selected_ref_id = new_name
    if st.session_state.selected_test_id == old_name:
        st.session_state.selected_test_id = new_name
    return True

def _clear_all():
    """Reset all project data."""
    st.session_state.profiles          = {}
    st.session_state.fit_results       = {}
    st.session_state.selected_ref_id   = None
    st.session_state.selected_test_id  = None
    st.session_state.bootstrap_results = None
    st.session_state.project_metadata  = {
        "name": "Untitled Project", "description": "",
        "created": "", "analyst": "",
    }
    st.session_state.active_substance  = {
        "name": "", "pubchem": None, "bcs_class": None,
        "fda_methods": [], "selected_method": None, "fetch_done": False,
    }

