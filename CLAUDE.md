# DissolvA — Project Guide & Handoff (read me first)

> This file travels with the repo so any machine/Claude session is fully oriented.
> Last updated: 2026-06-10 (free beta shipped: F1–F4 + value-adds; native nav
> palette + slim consent banner; ready for the school-distribution day).

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
  **LIVE on Railway: https://dissolva-backend-production-d2c7.up.railway.app**
  (Railway project `zoological-endurance`; deploys from `main`, `dev` = staging).
  The Streamlit secret `[backend] url` points here; `api_key`/`admin_key` match the
  Railway env `BACKEND_API_KEY`/`ADMIN_API_KEY`. Postgres plugin attached →
  membership + usage analytics live (Admin Console populated). Frontend calls it via
  `dissolva/engine_client.py` → **compute runs server-side**; `engine.py` mirrors
  `dissolva/models.py`. Same `dev`→`main` promote workflow.
  NOTE: a duplicate empty Railway project `giving-abundance` exists — ignore/delete it.

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

## Deploy workflow (two machines: work + home)
Daily loop: **work on `dev` → push `dev`** (syncs both machines; the `dev` Streamlit
app auto-deploys as a live preview). When a chunk is solid: **promote `dev`→`main`**
(`git push origin dev:main`) — this rebuilds production dissanalyze.streamlit.app +
Railway backend. Golden rule: **push before leaving a machine, `git checkout dev &&
git pull` before starting on the other** (avoids history divergence). HTTPS + PAT for
pushes. Streamlit Cloud caches: after a deploy, hard-refresh (Cmd+Shift+R) or Reboot
the app from the dashboard if it "looks old".

## Current status (2026-06-09)
- ✅ Live, all English. 62-model engine, fixed bootstrap f2 (FDA 85% rule),
  data_input data-corruption fixed, FDA CV early-point rule fixed.
- ✅ **Backend live + ACTIVATED** — compute runs on Railway; frontend falls back to
  local engine only if the backend is unreachable.
- ✅ **Google sign-in WORKS** via `streamlit-oauth` (`dissolva/auth.py`) — native
  `st.login` was unreliable on Community Cloud. Needs `[auth]`+`[auth.google]` secrets.
- ✅ **P0 hardening & polish done** (see plan `~/.claude/plans/witty-juggling-horizon.md`):
  backend input bounds (DoS), XSS-escape external content in `api_information.py`,
  Korsmeyer-Peppas ≤60% validity note, f2 3-point guard, fitting legend + chart DPI/
  contrast, sign-in button shrink + "Powered by AI" moved to footer fine print.
- 🚧 **IVIVC is DISABLED on purpose** (`pages/ivivc.py` shows a notice + `st.stop()`)
  — Level A misapplied Wagner-Nelson (used dissolution instead of plasma Cp →
  circular) and Multiple Level C regressed a constant (crash). Needs a proper
  rewrite: Level A from entered plasma Cp [Fa=(Cp+ke·AUC0-t)/(ke·AUC0-∞)],
  Multiple-C with real per-formulation dissolution values.

## Free-beta hardening progress (2026-06-09, on `dev` — promote `dev→main` to ship)
Plan: `~/.claude/plans/tender-kindling-dahl.md`. Decision: Friday = **free beta** (paid later);
beta gives everyone a free **core** account; owner needs an admin console + privacy-safe usage analytics.
- ✅ **F1** runtime crash fixes (excel_report safe metrics; kinetic_model nan-safe table/curve/params;
  bootstrap distribution guard; data_input ≥2-timepoint + non-monotonic checks). AppTest 0 exc.
- ✅ **F2** backend security: API-key (`X-API-Key` vs `BACKEND_API_KEY`) on /api/*; slowapi rate
  limits; prod fail-fast; tighter CORS/TrustedHost; request logging; bootstrap cap 5000. pytest 5/5.
  **Frontend** sends key from `st.secrets["backend"]["api_key"]`. security.py has `require_admin_key` ready.
- ✅ **F3** honest header (dropped IVIVC, "FDA/EMA guidance-aligned", BETA chip) + privacy dialog
  rewritten (discloses beta data: email/name, country, usage; reaffirms dissolution data never stored).
- ✅ **F4** Membership + analytics + admin console (code-complete, tested via SQLite).
  Backend: `db.py` (SQLAlchemy Member/UsageEvent; no-op when `DATABASE_URL` empty),
  `routes/members.py` (/api/members/upsert +geo country, /api/events, admin /api/admin/{members,stats}
  behind X-Admin-Key). Frontend: `auth.py` upserts a free 'core' member on sign-in;
  `engine_client` upsert_member/log_event (fire-and-forget) + admin_members/admin_stats;
  page-view events on nav change; hidden **Admin** page (`pages/admin.py`) for admin emails
  (`st.secrets[admin][emails]`, default owner). Privacy: only email/name/country/feature — no science data.

## Beta value-adds (2026-06-10, shipped to main) — `dissolva/extras.py`
Low-effort, high-impact additions for the academic launch. All logic is isolated in
`dissolva/extras.py`; each piece degrades gracefully and cannot crash the app.
- **Modern nav:** sidebar uses `streamlit-option-menu` (icons + amber pill) with a
  try/except **fallback to `st.radio`**; thin Setup→Analysis→Report→Reference
  **stepper** at the top of the main content (driven by the current page).
- **Load demo data** (sidebar): one click loads Reference + Test 6-vessel IR profiles
  and selects them → instant fitting/f2.
- **Cite this tool** (sidebar dialog): APA + BibTeX. Override via `[citation]` secret
  (`authors,title,year,version,url,doi`); add the Zenodo `doi` once minted.
- **Publication exports** (Excel Report page expander): PDF report (matplotlib
  PdfPages — cover + overlay + model ranking) + **300 dpi PNG** figure.
- **Sentry** crash reporting (frontend `app.py` + backend `main.py`): no-op unless a
  DSN is set; `send_default_pii=False`, `traces_sample_rate=0` → no user data.
- **GDPR consent banner** (session-level, dismissible).

### Optional secrets/env for the value-adds (all optional — app works without them)
- Frontend secrets: `[sentry] dsn="..."`; `[citation] doi="10.5281/zenodo.XXXX"` etc.
- Backend Railway env: `SENTRY_DSN=...` (separate project from the frontend).

### Deploy steps to activate the beta (all F1–F4)
1. Promote: in `~/dissolva/app` and `~/dissolva/backend`: `git push origin dev:main`.
2. **Railway (backend)**: add the **Postgres** plugin (auto-sets `DATABASE_URL`); set env
   `BACKEND_API_KEY=<random>`, `ADMIN_API_KEY=<random>`, strong `SECRET_KEY`, `ENVIRONMENT=production`.
3. **Streamlit Cloud (frontend) secrets**: under `[backend]` keep `url`, add
   `api_key="<BACKEND_API_KEY>"` and `admin_key="<ADMIN_API_KEY>"`; add `[admin]` `emails=["msinankaynak@gmail.com"]`.
4. Verify: sign in → you appear in the **Admin** console; usage events accrue.
   Without step 2/3 the API stays OPEN and membership/analytics no-op (app still works).

## Roadmap (next steps) — full detail in `~/.claude/plans/tender-kindling-dahl.md`
Goal: replace DDSolver/KinetDS; sell to formulation-development researchers worldwide.
- ✅ **DONE:** backend deploy + activation, frontend↔API wiring, Google auth, **P0**.
- **P1 — Security/IP (urgent, before monetizing):** backend API-key auth (`/api/*`
  are currently PUBLIC) + rate limiting (`slowapi`) + request logging; tighten CORS/
  TrustedHost; `SECRET_KEY` fail-fast; THEN **remove the engine from this PUBLIC repo**
  (`dissolva/models.py` heavy parts + `engine_client.py` local fallback) → real IP
  protection (engine still mirrored here as fallback).
- **P2 — Scientific parity (beat DDSolver/KinetDS):** weighted least squares
  (`sigma=`), per-parameter 95% CIs, residual diagnostics (residual/Q-Q/runs), dF/dt
  + Tx% + PDTS/MSD, richer export.
- **P3 — Commercial:** Stripe (Payment Link + Portal), wire `state.require_tier()`
  (defined but unused), Firestore persistence/usage metering, ToS/Privacy Policy,
  fix "FDA compliant" marketing claim, 21 CFR Part 11 audit trail.
- **P4 — IVIVC rewrite** (plasma-Cp Level A; see below) + validation suite vs
  `bootf2`/`disprofas`/DDSolver to make the site's "8/8 validation" claim auditable.
- **api_information.py** further hardening: a few secondary external fields still
  unescaped; graceful network-failure messages; disclose scite.ai (commercial).

## Notes / gotchas
- The repo was made PUBLIC to deploy on Streamlit Cloud free (private repo +
  Streamlit free was unworkable). Real IP protection = backend separation (above).
- Streamlit subdomains can't contain "sso" → "dissolva"/"dissolution" are blocked,
  hence `dissanalyze`.
- Privacy truth (for the notice): no dissolution data leaves the machine/server;
  only the drug name typed on API Information goes to PubChem/FDA/PubMed/scite.ai.
