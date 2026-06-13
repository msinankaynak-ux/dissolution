"""DissolvA page: My Records (21 CFR Part 11 compliance mode).

Server-side, audit-trailed dissolution records. Saving/loading goes through the
backend API (engine_client) which writes an immutable, hash-chained audit log.
Falls back to an informational notice when the backend is not configured.
UI strings are i18n-wrapped (EN/TR)."""

import json

import streamlit as st

from dissolva import auth, engine_client, i18n

_t = i18n.t
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
    st.header(_t("My Records"))
    st.caption(
        _t(
            "Compliance mode — your analyses are saved server-side with an immutable, "
            "time-stamped audit trail (21 CFR Part 11 readiness). Records stay private to your account."
        )
    )

    if not engine_client.records_enabled():
        st.info(
            _t(
                "**Compliance records are not enabled on this deployment.**\n\n"
                "Saving/loading with an audit trail needs the backend configured "
                '(`st.secrets["backend"]`). Until then, your work is kept locally in your '
                "browser (autosave) only."
            )
        )
        return

    if not auth.is_authenticated():
        st.warning(
            _t(
                "Sign in (sidebar) to save and load your records — records are tied to your account."
            )
        )
        return

    owner = (auth.current_user() or {}).get("email") or ""
    if not owner:
        st.warning(_t("Could not determine your account email; please sign in again."))
        return

    # ── Audit trail status ───────────────────────────────────────────────────
    intact = engine_client.audit_chain_intact()
    if intact is True:
        st.success(_t("Audit trail: **intact** ✅ (hash-chain verified)"))
    elif intact is False:
        st.error(
            _t(
                "Audit trail: **TAMPER DETECTED** ⚠️ — the audit log hash-chain does not verify."
            )
        )
    else:
        st.caption(_t("Audit trail status: unavailable."))

    # ── Save current work ────────────────────────────────────────────────────
    st.subheader(_t("Save current work"))
    st.caption(
        _t(
            "💡 Your work is auto-kept in your browser so you never lose it. **Saving here** "
            "writes a permanent, audit-trailed record on the server — nothing is written to your "
            "records until you click Save/Update."
        )
    )
    _active_id = st.session_state.get("_active_record_id")
    _active_name = st.session_state.get("_active_record_name") or _t("this record")

    if not st.session_state.get("profiles"):
        st.caption(_t("Load or enter dissolution profiles first, then save them here."))
    else:
        if _active_id:
            st.markdown(
                "<div style='color:#9fb0d0;font-size:0.86rem;margin-bottom:6px;'>"
                + _t("📂 Editing saved record:")
                + " <b style='color:#FFCC00;'>"
                + _active_name
                + "</b></div>",
                unsafe_allow_html=True,
            )
            u1, u2 = st.columns(2)
            if u1.button(
                _t("⬆️ Update “{name}”").format(name=_active_name),
                type="primary",
                use_container_width=True,
                key="rec_update_btn",
                help=_t(
                    "Overwrite the loaded record (saves a new version; the previous "
                    "version stays in the audit trail)."
                ),
            ):
                try:
                    engine_client.save_record(
                        owner, _active_name, _current_payload(), analysis_id=_active_id
                    )
                    st.toast(_t("Record updated — new version saved."), icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(_t("Update failed: {e}").format(e=e))
            if u2.button(
                _t("➕ Save as a new record"),
                use_container_width=True,
                key="rec_saveas_btn",
            ):
                st.session_state["_show_saveas"] = True

        if (not _active_id) or st.session_state.get("_show_saveas"):
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            with c1:
                _default_name = st.session_state.get("project_metadata", {}).get(
                    "name", "Untitled Project"
                )
                rec_name = st.text_input(
                    _t("New record name"), value=_default_name, key="rec_name_in"
                )
            with c2:
                if st.button(
                    _t("💾 Save new"),
                    type="primary",
                    use_container_width=True,
                    key="rec_save_btn",
                ):
                    try:
                        res = engine_client.save_record(
                            owner, rec_name or "Untitled", _current_payload()
                        )
                        st.session_state["_active_record_id"] = res.get("id")
                        st.session_state["_active_record_name"] = rec_name or "Untitled"
                        st.session_state["_show_saveas"] = False
                        st.toast(_t("Saved as a new record."), icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(_t("Save failed: {e}").format(e=e))

    st.divider()

    # ── Saved records ────────────────────────────────────────────────────────
    st.subheader(_t("Saved records"))
    try:
        records = engine_client.list_records(owner)
    except Exception as e:
        st.error(_t("Could not load your records: {e}").format(e=e))
        return

    if not records:
        st.caption(_t("No saved records yet."))
        return

    for r in records:
        with st.container(border=True):
            cc1, cc2, cc3 = st.columns([4, 1, 1])
            _meta = _t("v{v} · updated {d}").format(
                v=r.get("version", 1),
                d=r.get("updated_at", "")[:19].replace("T", " "),
            )
            cc1.markdown(
                f"**{r.get('name','Untitled')}**  \n"
                f"<span style='color:#9fb0d0;font-size:0.78rem;'>{_meta}</span>",
                unsafe_allow_html=True,
            )
            if cc2.button(
                _t("Load"), key=f"rec_load_{r['id']}", use_container_width=True
            ):
                try:
                    full = engine_client.get_record(owner, r["id"])
                    if _hydrate(full.get("payload")):
                        st.session_state["_active_record_id"] = r["id"]
                        st.session_state["_active_record_name"] = r.get("name", "")
                        st.session_state["_show_saveas"] = False
                        st.toast(
                            _t("Loaded “{name}”.").format(name=r.get("name", "")),
                            icon="📂",
                        )
                        st.rerun()
                    else:
                        st.error(_t("Could not read this record's data."))
                except Exception as e:
                    st.error(_t("Load failed: {e}").format(e=e))
            if cc3.button(
                _t("Delete"), key=f"rec_del_{r['id']}", use_container_width=True
            ):
                st.session_state[f"_confirm_del_{r['id']}"] = True
            if st.session_state.get(f"_confirm_del_{r['id']}"):
                rsn = st.text_input(
                    _t("Reason for deletion (kept in audit trail)"),
                    key=f"rec_delrsn_{r['id']}",
                )
                dc1, dc2 = st.columns(2)
                if dc1.button(
                    _t("Confirm delete"), type="primary", key=f"rec_delok_{r['id']}"
                ):
                    try:
                        engine_client.delete_record(owner, r["id"], reason=rsn or "")
                        if r["id"] == st.session_state.get("_active_record_id"):
                            st.session_state.pop("_active_record_id", None)
                            st.session_state.pop("_active_record_name", None)
                        st.session_state.pop(f"_confirm_del_{r['id']}", None)
                        st.toast(_t("Record archived (soft-deleted)."), icon="🗑️")
                        st.rerun()
                    except Exception as e:
                        st.error(_t("Delete failed: {e}").format(e=e))
                if dc2.button(_t("Cancel"), key=f"rec_delcancel_{r['id']}"):
                    st.session_state.pop(f"_confirm_del_{r['id']}", None)
                    st.rerun()

    # ── Audit trail viewer ───────────────────────────────────────────────────
    st.divider()
    with st.expander(_t("🔎 Audit trail — your activity (who · what · when)")):
        try:
            _entries = engine_client.list_audit(owner)
        except Exception as _e:
            st.caption(_t("Audit trail unavailable: {e}").format(e=_e))
            _entries = []
        if not _entries:
            st.caption(_t("No audit entries yet."))
        else:
            import pandas as _pd

            _adf = _pd.DataFrame(
                [
                    {
                        _t("When (UTC)"): (e.get("ts", "") or "")[:19].replace(
                            "T", " "
                        ),
                        _t("Action"): e.get("action", ""),
                        _t("Record"): (e.get("entity_id", "") or "")[:8],
                    }
                    for e in _entries
                ]
            )
            st.dataframe(_adf, use_container_width=True, hide_index=True)
            st.caption(
                _t(
                    "Immutable, hash-chained log (21 CFR Part 11 §11.10(e)) — append-only, "
                    "cannot be edited."
                )
            )
