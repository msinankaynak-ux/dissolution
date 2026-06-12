"""Admin console (hidden) — members + privacy-safe usage analytics.

Only rendered for admin emails (gated in app.py). Reads from the backend admin
endpoints via engine_client (X-Admin-Key). No scientific data is ever shown here.
"""
import streamlit as st
import pandas as pd

from dissolva import engine_client


def render():
    st.header("🛡️ Admin Console")
    st.caption("Members and privacy-safe usage analytics (account/country/feature only — "
               "never any dissolution data).")

    if not engine_client.using_backend():
        st.warning("Backend not configured — nothing to show.")
        return

    try:
        stats = engine_client.admin_stats()
        members = engine_client.admin_members()
    except Exception as e:
        st.error(f"Could not load admin data ({type(e).__name__}). "
                 f"Check the backend URL and the admin key in secrets.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Total members", stats.get("total_members", 0))
    c2.metric("Usage events", stats.get("total_events", 0))

    st.subheader("Members")
    ms = members.get("members", [])
    if ms:
        df = pd.DataFrame(ms)
        if "tier" in df.columns:
            from dissolva import tiers as _t
            df["tier"] = df["tier"].map(lambda x: _t.TIERS.get(_t.normalize_tier(x), {}).get("label", str(x)))
        # Online indicator: green dot if last_seen is within the last 5 minutes (server UTC).
        import datetime as _dt
        def _is_online(ls):
            try:
                return (_dt.datetime.utcnow() - _dt.datetime.fromisoformat(str(ls))).total_seconds() < 300
            except Exception:
                return False
        if "last_seen" in df.columns:
            _onl = df["last_seen"].map(_is_online)
            df.insert(0, "•", _onl.map(lambda o: "🟢" if o else "⚪"))
            st.caption(f"🟢 {int(_onl.sum())} online now · {len(df)} total (active = last_seen within 5 min)")
        cols = [c for c in ["•", "email", "name", "country", "tier", "created_at", "last_seen"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No members yet.")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("By country")
        bc = stats.get("by_country", [])
        if bc:
            st.bar_chart(pd.DataFrame(bc).set_index("label")["count"])
        else:
            st.caption("No data yet.")
    with cc2:
        st.subheader("Top features")
        tf = stats.get("top_features", [])
        if tf:
            st.bar_chart(pd.DataFrame(tf).set_index("label")["count"])
        else:
            st.caption("No data yet.")
