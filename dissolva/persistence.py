"""DissolvA — client-side persistence (Faz 1).

Aktif projeyi TARAYICININ localStorage'ına kaydeder; böylece sayfa yenilenince/
bağlantı kopunca kullanıcının çalışması SİLİNMEZ ve "aynı cihazda yarın devam"
çalışır.

GİZLİLİK (kritik): Veri YALNIZCA ziyaretçinin kendi tarayıcısında tutulur.
DissolvA sunucusuna asla gönderilmez/saklanmaz. Bu, projenin "we have no database
of your scientific data" vaadini korur. Ancak "erased on refresh" ifadesi artık
geçerli olmadığından gizlilik metni güncellenmelidir (bkz. uygulama rehberi).

Free katman = tam olarak BİR yerel kayıtlı proje (aktif olan). Çok-proje /
çok-cihaz senkron, Pro'nun sunucu-tabanlı özelliğidir (Faz 2).

Hedef konum (repoda): dissolva/persistence.py
"""
from __future__ import annotations
import json
import os
import numpy as np
import streamlit as st

try:
    from streamlit_local_storage import LocalStorage
    _LS_OK = True
except Exception:
    _LS_OK = False

# Kill-switch: gerekirse (ör. headless test / sorun giderme) localStorage'i
# tamamen kapatmak icin DISSOLVA_DISABLE_LOCALSTORAGE=1. App her durumda calisir.
if os.environ.get("DISSOLVA_DISABLE_LOCALSTORAGE") == "1":
    _LS_OK = False

_LS_KEY = "dissolva_active_project_v1"

# "Aktif proje"yi oluşturan session_state anahtarları. fit_results ve
# bootstrap_results BİLEREK saklanmaz — profillerden yeniden hesaplanır
# (payload'ı küçük tutar, bayat sonuç riskini önler).
PERSIST_KEYS = [
    "profiles",
    "method_cfg",
    "active_substance",
    "project_metadata",
    "selected_ref_id",
    "selected_test_id",
]


def _ls():
    """Run başına tek LocalStorage örneği (session_state'te önbelleklenir)."""
    if not _LS_OK:
        return None
    if "_ls_obj" not in st.session_state:
        st.session_state["_ls_obj"] = LocalStorage()
    return st.session_state["_ls_obj"]


def _jsonable(obj):
    """numpy tip/dizilerini JSON-güvenli Python tiplerine çevirir (özyinelemeli)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _snapshot() -> dict:
    return {k: _jsonable(st.session_state.get(k)) for k in PERSIST_KEYS
            if k in st.session_state}


def restore_on_load():
    """localStorage'tan session_state'i tarayıcı oturumunda BİR KEZ doldurur,
    yalnızca henüz devam eden bir çalışma yoksa. init_session_state()'ten hemen
    sonra çağır.

    Not: streamlit-local-storage bileşeni ilk okumada hazır olmayabilir; bu yüzden
    birkaç deneme hakkı tanıyoruz (rerun handshake)."""
    if not _LS_OK:
        return
    if st.session_state.get("_ls_restored"):
        return
    # Zaten veri olan oturumu ezme (ör. demo verisi yeni yüklendi).
    if st.session_state.get("profiles"):
        st.session_state["_ls_restored"] = True
        return
    raw = None
    try:
        raw = _ls().getItem(_LS_KEY)
    except Exception:
        raw = None
    if raw:
        try:
            data = json.loads(raw)
            for k, v in data.items():
                if k in PERSIST_KEYS:
                    st.session_state[k] = v
            st.session_state["_ls_had_saved"] = True
        except Exception:
            pass
        st.session_state["_ls_restored"] = True
        return
    # Bileşen henüz hazır değilse birkaç rerun daha bekle, sonra kilitle.
    _att = st.session_state.get("_ls_restore_attempts", 0)
    if _att < 3:
        st.session_state["_ls_restore_attempts"] = _att + 1
    else:
        st.session_state["_ls_restored"] = True


def autosave():
    """Aktif projeyi localStorage'a yazar. Script run'ının SONUNDA bir kez çağır.
    Demo modunda no-op (örnek veri kullanıcının çalışması değildir)."""
    if not _LS_OK:
        return
    if st.session_state.get("demo_mode"):
        return
    try:
        payload = json.dumps(_snapshot(), ensure_ascii=False)
        _ls().setItem(_LS_KEY, payload, key="_ls_set_active")
    except Exception:
        pass


def clear_local():
    """Yerel kayıtlı projeyi siler (New Session / Clear saved data'dan çağrılır)."""
    if not _LS_OK:
        return
    try:
        _ls().deleteItem(_LS_KEY)
    except Exception:
        pass


def has_local_save() -> bool:
    """Bu oturumda localStorage'tan veri geri yüklendiyse True."""
    return bool(st.session_state.get("_ls_had_saved"))
