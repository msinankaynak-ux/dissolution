"""DissolvA page: My Records (21 CFR Part 11 compliance mode).

Server-side, audit-trailed dissolution records. Saving/loading goes through the
backend API (engine_client) which writes an immutable, hash-chained audit log.
Falls back to an informational notice when the backend is not configured."""

import json

import streamlit as st

from dissolva import auth, engine_client

_PERSIST_KEYS = ["profiles", "method_cfg", "active_substance", "project_metadata"]


def _current_payload():
    return {k: st.session_state.get(k) for k in _PERSIST_KEYS if k in st.session_state}


def _hydrate(payload):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return False
    for k in _PERSIST_KEYS:
        if k in payload and payload[k] is not None:
            st.session_state[k] = payload[k]
    return True


def render():
    st.header("My Records")
    st.caption(
        "Compliance mode — your analyses are saved server-side with an immutable, "
        "time-stamped audit trail (21 CFR Part 11 readiness). Records stay private to your account."
    )

    if not engine_client.records_enabled():
        st.info(
            "**Compliance records are not enabled on this deployment.**\n\n"
            "Saving/loading with an audit trail needs the backend configured "
            '(`st.secrets["backend"]`). Until then, your work is kept locally in your '
            "browser (autosave) only."
        )
        return

    if not auth.is_authenticated():
        st.warning(
            "Sign in (sidebar) to save and load your records — records are tied to your account."
        )
        return

    owner = (auth.current_user() or {}).get("email") or ""
    if not owner:
        st.warning("Could not determine your account email; please sign in again.")
        return

    # ── Audit trail status ───────────────────────────────────────────────────
    intact = engine_client.audit_chain_intact()
    if intact is True:
        st.success("Audit trail: **intact** ✅ (hash-chain verified)")
    elif intact is False:
        st.error(
            "Audit trail: **TAMPER DETECTED** ⚠️ — the audit log hash-chain does not verify."
        )
    else:
        st.caption("Audit trail status: unavailable.")

    # ── Save current work ────────────────────────────────────────────────────
    st.subheader("Save current work")
    if not st.session_state.get("profiles"):
        st.caption("Load or enter dissolution profiles first, then save them here.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            rec_name = st.text_input(
                "Record name",
                value=st.session_state.get("project_metadata", {}).get(
                    "name", "Untitled Project"
                ),
                key="rec_name_in",
            )
        with c2:
            st.write("")
            st.write("")
            if st.button(
                "💾 Save", type="primary", use_container_width=True, key="rec_save_btn"
            ):
                try:
                    res = engine_client.save_record(
                        owner, rec_name or "Untitled", _current_payload()
                    )
                    st.session_state["_active_record_id"] = res.get("id")
                    st.toast("Saved to your compliance records.", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

    st.divider()

    # ── Saved records ────────────────────────────────────────────────────────
    st.subheader("Saved records")
    try:
        records = engine_client.list_records(owner)
    except Exception as e:
        st.error(f"Could not load your records: {e}")
        return

    if not records:
        st.caption("No saved records yet.")
        return

    for r in records:
        with st.container(border=True):
            cc1, cc2, cc3 = st.columns([4, 1, 1])
            cc1.markdown(
                f"**{r.get('name','Untitled')}**  \n"
                f"<span style='color:#9fb0d0;font-size:0.78rem;'>v{r.get('version',1)} · "
                f"updated {r.get('updated_at','')[:19].replace('T',' ')}</span>",
                unsafe_allow_html=True,
            )
            if cc2.button("Load", key=f"rec_load_{r['id']}", use_container_width=True):
                try:
                    full = engine_client.get_record(owner, r["id"])
                    if _hydrate(full.get("payload")):
                        st.session_state["_active_record_id"] = r["id"]
                        st.toast(f"Loaded “{r.get('name','')}”.", icon="📂")
                        st.rerun()
                    else:
                        st.error("Could not read this record's data.")
                except Exception as e:
                    st.error(f"Load failed: {e}")
            if cc3.button("Delete", key=f"rec_del_{r['id']}", use_container_width=True):
                st.session_state[f"_confirm_del_{r['id']}"] = True
            if st.session_state.get(f"_confirm_del_{r['id']}"):
                rsn = st.text_input(
                    "Reason for deletion (kept in audit trail)",
                    key=f"rec_delrsn_{r['id']}",
                )
                dc1, dc2 = st.columns(2)
                if dc1.button(
                    "Confirm delete", type="primary", key=f"rec_delok_{r['id']}"
                ):
                    try:
                        engine_client.delete_record(owner, r["id"], reason=rsn or "")
                        st.session_state.pop(f"_confirm_del_{r['id']}", None)
                        st.toast("Record archived (soft-deleted).", icon="🗑️")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                if dc2.button("Cancel", key=f"rec_delcancel_{r['id']}"):
                    st.session_state.pop(f"_confirm_del_{r['id']}", None)
                    st.rerun()
