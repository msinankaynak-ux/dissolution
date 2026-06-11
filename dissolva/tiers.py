"""DissolvA tiers — single source of truth for plans, feature entitlement and
beta state.

During the free beta, every LIVE feature is unlocked for everyone (BETA=True);
tiers are still recorded (analytics) and shown so the value ladder is clear, and
pricing activates at launch. Enterprise is sales-led (contact), not self-serve.

Pure data + helpers (no Streamlit import) so it is safe to import anywhere.
"""

# Global beta switch. True = all LIVE features free for everyone; pricing shown
# as "free during beta · paid at launch". Flip to False at commercial launch.
BETA = True

# ── Tier ladder ────────────────────────────────────────────────────────────
TIERS = {
    "free": {
        "key": "free", "label": "Free", "rank": 0, "color": "#9fb0d0",
        "audience": "Students & academic researchers",
        "price": "$0", "price_note": "Free forever",
        "cta": "Get started", "cta_type": "free",
    },
    "pro": {
        "key": "pro", "label": "Pro", "rank": 1, "color": "#FFCC00",
        "audience": "Industry R&D scientists, consultants, small labs",
        "price": "$19/mo", "price_note": "Free during beta · paid at launch",
        "cta": "Join Pro beta", "cta_type": "free",
    },
    "enterprise": {
        "key": "enterprise", "label": "Enterprise", "rank": 2, "color": "#5dd0ff",
        "audience": "Pharma / generic manufacturers, CROs",
        "price": "Custom", "price_note": "Annual / per-seat · contact us",
        "cta": "Contact us", "cta_type": "contact",
    },
}
TIER_ORDER = ["free", "pro", "enterprise"]
TIER_RANK = {k: v["rank"] for k, v in TIERS.items()}

# Legacy session / backend tier keys → canonical (core→free, research→pro, pro→enterprise).
_LEGACY = {"core": "free", "research": "pro", "pro": "enterprise"}


def normalize_tier(t):
    if t in TIERS:
        return t
    return _LEGACY.get(t, "free")


# ── Feature → tier + build status ────────────────────────────────────────────
# status: "live" (built) or "soon" (planned/roadmap). During beta, LIVE features
# are free for everyone; SOON features are shown as "coming soon" regardless of tier.
FEATURES = {
    # Free
    "kinetic_models":   {"tier": "free", "status": "live", "label": "62 kinetic models + fitting"},
    "f1_f2":            {"tier": "free", "status": "live", "label": "f1/f2 similarity + USP/FDA/EMA compliance"},
    "statistics":       {"tier": "free", "status": "live", "label": "Statistics — MDT, DE, RSD"},
    "bootstrap_param":  {"tier": "free", "status": "live", "label": "Bootstrap f2 (parametric)"},
    "template_builder": {"tier": "free", "status": "live", "label": "Template Builder + example data"},
    "academy":          {"tier": "free", "status": "live", "label": "Academy — models & methods"},
    "excel_basic":      {"tier": "free", "status": "live", "label": "Excel report (research-use watermark)"},
    # Pro
    "excel_full":       {"tier": "pro", "status": "live", "label": "Full 8-sheet Excel report, no watermark"},
    "bootstrap_bca":    {"tier": "pro", "status": "live", "label": "Bootstrap f2 — nonparametric / BCa"},
    "batch":            {"tier": "pro", "status": "soon", "label": "Batch (multi-profile) + comparative PDF"},
    "ai_reco":          {"tier": "pro", "status": "soon", "label": "AI model recommendation + regulatory guidance"},
    "ivivc":            {"tier": "pro", "status": "soon", "label": "IVIVC module"},
    "branding":         {"tier": "pro", "status": "soon", "label": "Custom branding on reports"},
    # Enterprise
    "part11":           {"tier": "enterprise", "status": "soon", "label": "21 CFR Part 11 — audit trail + e-signatures"},
    "validation":       {"tier": "enterprise", "status": "soon", "label": "Software validation package (IQ/OQ/PQ)"},
    "team_sso":         {"tier": "enterprise", "status": "soon", "label": "Team seats + admin console + SSO"},
    "deploy":           {"tier": "enterprise", "status": "soon", "label": "On-prem / private deployment + SLA support"},
}


def feature_state(key):
    """'available' (usable now), 'soon' (planned), or 'locked' (post-beta gated)."""
    f = FEATURES.get(key)
    if not f:
        return "available"
    if f["status"] == "soon":
        return "soon"
    return "available" if BETA else "locked"


def plans():
    """Tier cards with their feature lists (for rendering a plans surface)."""
    out = []
    for tk in TIER_ORDER:
        t = dict(TIERS[tk])
        t["features"] = [
            {"label": f["label"], "status": f["status"]}
            for f in FEATURES.values() if f["tier"] == tk
        ]
        out.append(t)
    return out
