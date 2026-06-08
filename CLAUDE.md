# DissolvA — Project Guide & Handoff (read me first)

> This file travels with the repo so any machine/Claude session is fully oriented.
> Last updated: 2026-06-09 (home machine session).

## ⚠️ Language rule (always)
**All user-facing app text MUST be in English** — every UI label, button,
`st.*` message, expander title, metric, chart title/axis, `help=` text. Audience
is international (FDA/EMA). Code comments, commits, and chat may be Turkish; only
end-user-visible strings must be English.

## What this is
DissolvA — a Streamlit dissolution-kinetics analysis app for pharmacists.
Engine: `dissolva/models.py` (62 kinetic models, RMSE/AICc/BIC metrics, bounded
curve_fit, f1/f2, vessel-level bootstrap f2). Pages: `dissolva/pages/`.

## Where everything lives
- **Frontend (this repo):** `github.com/msinankaynak-ux/dissolution` — **PUBLIC**.
  `main` = production, `dev` = development. Live at **dissanalyze.streamlit.app**
  (Streamlit Community Cloud, deploys from `main`, Python 3.11). Landing page
  `dissolva.app` (repo `dissolva-website`) links to it via "Launch App".
- **Backend:** `github.com/msinankaynak-ux/dissolva-backend` — **PRIVATE**.
  FastAPI engine API (`/api/fit`, `/api/f2`, `/api/bootstrap-f2`, `/api/models`).
  Built + tested (5/5), Railway-ready (Dockerfile binds `$PORT`). NOT deployed yet.

## Run locally
```bash
cd <repo>/app   # or wherever this repo is cloned
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Verify a change headlessly (no browser):
```bash
python -c "from streamlit.testing.v1 import AppTest; \
  print('exceptions:', len(AppTest.from_file('app.py').run().exception))"
```

## Deploy workflow
Develop on `dev`, then promote to production: `git push origin dev:main`
(this triggers a rebuild of dissanalyze.streamlit.app). Keep `main` == `dev` tree.

## Current status (2026-06-09)
- ✅ Live, all English. 62-model engine, fixed bootstrap f2 (FDA 85% rule),
  data_input data-corruption fixed, FDA CV early-point rule fixed.
- ✅ Data-privacy notice (sidebar button → dialog). Google OIDC scaffold in
  `dissolva/auth.py` (open mode unless `.streamlit/secrets.toml` `[auth]` set).
- 🚧 **IVIVC is DISABLED on purpose** (`pages/ivivc.py` shows a notice + `st.stop()`)
  — Level A misapplied Wagner-Nelson (used dissolution instead of plasma Cp →
  circular) and Multiple Level C regressed a constant (crash). Needs a proper
  rewrite: Level A from entered plasma Cp [Fa=(Cp+ke·AUC0-t)/(ke·AUC0-∞)],
  Multiple-C with real per-formulation dissolution values.

## Open tasks (next steps)
1. **Deploy backend to Railway** (needs your Railway account; see backend repo
   `DEPLOY_RAILWAY.md`). Get the URL.
2. **Wire frontend → backend API** (Claude can do fully): pages call the API
   instead of importing `dissolva.models`; then remove the engine from this
   PUBLIC repo → real IP protection.
3. **Google login** is PARKED: secrets loaded (button shows) but clicking gave
   "internal server error" — likely wrong client_secret or consent-screen
   (Testing → add test user / publish). Diagnose from the app logs. Optional for beta.
4. **api_information.py hardening** (review found): html.escape external content
   (XSS), graceful network-failure messages, BCS-attribution confidence, regex
   for FDA speed/volume, disclose scite.ai (commercial) in privacy note.

## Notes / gotchas
- The repo was made PUBLIC to deploy on Streamlit Cloud free (private repo +
  Streamlit free was unworkable). Real IP protection = backend separation (above).
- Streamlit subdomains can't contain "sso" → "dissolva"/"dissolution" are blocked,
  hence `dissanalyze`.
- Privacy truth (for the notice): no dissolution data leaves the machine/server;
  only the drug name typed on API Information goes to PubChem/FDA/PubMed/scite.ai.
