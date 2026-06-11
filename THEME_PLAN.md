# DissolvA — Theming System: Plan & Handoff (read me, then continue)

> Purpose: hand the **3-theme** plan to a fresh Cowork / Claude Code session.
> **Dark theme is DONE and live.** Light + Hybrid + the theme switcher are **deferred** — this doc
> is the spec + how-to so they can be built later without surprises.

---

## 1. Where we are
- The app (`dissolution-app`, Streamlit) was refactored to a **premium Dark Mode** matching an approved mockup.
- Shipped to **production** (`dissanalyze.streamlit.app`, branch `main`) and `dev` — both at the same commit.
- The original interface brief asked for **3 themes (Dark / Light / Hybrid) + a theme switcher**, but said
  *"focus entirely on Dark for now; disregard light/hybrid/switcher."* So we built **Dark only**. The rest is below.
- Engine/math/state were NOT touched by the theming work (only layout + CSS + colors).

## 2. The 3 themes (interpretation we settled on)
The cleanest mapping = sidebar tone × workspace tone:

| Theme | Sidebar | Workspace | Status | Notes |
|---|---|---|---|---|
| **Dark** | `#0B132B` navy | `#1C2541` graphite | ✅ DONE (live) | Gold `#FFCC00` accent, white headers, `#CBD5E1` secondary text |
| **Hybrid** | dark `#0B132B` | **light** cream/white | ⏸️ deferred | This is basically the app's ORIGINAL look (dark sidebar + light content) |
| **Light** | light `#F5F0E8`/white | light | ⏸️ deferred | Fully light; needs dark text + light-surface cards |
| **Switcher** | — | — | ⏸️ deferred | Let the user pick the theme at runtime |

## 3. Design system
**Shared brand:** wordmark = `Dissolv` (theme's primary text color) + **`A` always gold `#FFCC00`** + `™`.
Logo icon = blue `#003171` square, gold `A`, gold corner notch. Accent gold `#FFCC00` everywhere
(charts use the older `#FFBF00` amber via `theme.AMBER` — leave charts as-is).

**Dark palette (implemented — see `dissolva/theme.py` constants):**
`NAVY_SIDEBAR=#0B132B · GRAPHITE=#1C2541 · SURFACE=#16203F · GOLD=#FFCC00 · TXT=#FFFFFF · TXT2=#CBD5E1`
config.toml: `base="dark", backgroundColor="#1C2541", secondaryBackgroundColor="#16203F",
primaryColor="#FFCC00", textColor="#E6ECF8"`.

**Light palette (proposed):** workspace `#F5F0E8`/white, surface white, text `#1a1a2e`/`#002147`,
secondary `#5a6480`, sidebar light `#FBF7EF` (or keep navy for Hybrid), accent gold `#FFCC00`.

**Hybrid palette (proposed = original look):** dark sidebar `#001a3d`/`#0B132B` + light workspace `#F5F0E8`
+ dark workspace text. (Git history before the dark-mode commits shows exactly this look.)

## 4. How Dark was built (the PATTERN to replicate per theme)
1. **`.streamlit/config.toml`** `base` + palette → native widgets (inputs, selects, dataframes, metrics,
   expanders, tabs, info/warning) auto-recolor. **This is the big lever.** ⚠️ `base` is set at STARTUP — see §5.
2. **`dissolva/theme.py`** — `_CSS` (sidebar, inputs, buttons, eq/info/step boxes, nav, oauth iframe scale) +
   color constants + `style_ax()` (charts kept light = clean light cards on dark; zero chart-text risk).
3. **Sidebar nav** (`app.py`) — categorized **button-nav** in `st.container(key="navmenu")`: 3 caps categories +
   `st.button(icon=":material/…:", type="primary"|"secondary")`; active = gold-glow via CSS scoped to
   `.st-key-navmenu`. Routing VALUES unchanged; display labels renamed via the category tuples.
4. **Header** (`app.py`) — `st.columns([0.56,0.44])`: left = title/tagline/BETA; right =
   `st.container(horizontal=True, horizontal_alignment="right")` with New Session (modal dialog), Load demo,
   `auth.render_sidebar_auth()`, Try Free. Breadcrumb chips recolored.
5. **Monetization** — sidebar bottom **Unlock Pro** gold-gradient card + Go Pro; page-bottom footer with
   version + Share Feedback + **Try DissolvA Pro — Upgrade Now** (placeholders, `href="#pro"`, until Stripe).
6. **Content sweep** — per page, flip light cards (`#f8f9fa`/`white`) → dark `#16203F` and dark text → light;
   KEEP gradient cards, the molecule-image white plate, and semantic colored badges. Worst page: `api_information.py`.

### ⚠️ Gotchas discovered (will bite the next theme too)
- **Streamlit STRIPS `!important` from inline styles.** To force a color over the sidebar `*` override, use a
  **class + `<style>` rule** (e.g. `.dvlogo-gold`), not inline `!important`. (This is why the logo A needed classes.)
- **`section[data-testid="stSidebar"] * { color … !important }`** silvers ALL custom sidebar colors — any
  colored sidebar element (logo, Unlock Pro card) needs class-based overrides.
- **Material icon size**: a `* {font-size}` rule shrinks the button icon too — size the label (`[data-testid="stMarkdownContainer"] p`) and the icon (`[data-testid="stIconMaterial"]`) SEPARATELY; add a `gap` for icon↔label spacing.
- **Local Streamlit was 1.58-incompatible** (Python 3.9) → we **pinned `streamlit==1.50.0`** so cloud == local
  test env. Without that, 1.58 rendered the nav centered while 1.50 was left-aligned ("works locally, broken on cloud").

## 5. The theme SWITCHER — the genuinely hard part (read before building)
Streamlit's `config.toml [theme] base` is read at **process startup** — it **cannot be changed at runtime** in
1.50. So a runtime switcher can't flip native widgets between dark/light bases live. Options:
- **(A) CSS-variable refactor (recommended):** rewrite `theme.py` `_CSS` to use CSS custom properties
  (`--bg, --surface, --txt, --txt2, --gold`) on a body wrapper class (`theme-dark|theme-light|theme-hybrid`),
  set from `st.session_state["theme"]`. All CUSTOM content + most overrides switch instantly. Native widgets
  follow the single config `base` — pick `base="dark"` and use CSS to lighten surfaces for light/hybrid (works
  for most widgets; test dataframes/inputs carefully). Content-sweep colors must also become variables.
- **(B) Reload trick:** store theme in a query param, and on change write a per-user config + `st.rerun()` — hacky,
  avoid.
- **Recommendation:** do (A). Budget it as ~the same effort as the dark build, because every hardcoded color
  (incl. the `api_information.py` sweep) must become a variable, ×3 themes.
- **Switcher UI:** a 3-way segmented control (st.segmented_control / radio) in the header-right or sidebar top,
  bound to `st.session_state["theme"]`, default `dark`.

## 6. Key files
- `dissolution-app/.streamlit/config.toml` — theme base + palette
- `dissolution-app/dissolva/theme.py` — `_CSS`, color constants, `style_ax`
- `dissolution-app/app.py` — sidebar brand + nav (`.st-key-navmenu`), header columns, breadcrumb, Unlock Pro, footer
- `dissolution-app/dissolva/pages/*.py` — per-page content colors (sweep target; `api_information.py` heaviest)
- Mockup reference: re-render via the chat (the v2 "complete dark workspace" mockup) or this repo's screenshots.

## 7. Constraints (do not break)
- **`streamlit==1.50.0` is pinned** (test==prod on the Py3.9 dev machine). Don't bump it without a Py3.10+ test env.
- **No engine/math/state changes** — theming is layout + CSS + colors only.
- **All user-facing text in English.**
- **Git: work on `dev` → push → "promote dev→main" only when solid.** Push before leaving a machine, pull before
  starting. Rebasing onto `origin/dev` keeps the other machine's commits intact (clean fast-forward pull).

## 8. Cowork starting prompt (paste this to continue the themes)
> Continue the DissolvA UI theming. Dark mode is DONE and live (see THEME_PLAN.md in the repo root). Next:
> build the **theme switcher + Light + Hybrid themes** per §5 (CSS-variable approach). First read THEME_PLAN.md
> and `dissolva/theme.py`, then propose the CSS-variable refactor plan (one variable set, swapped by a body
> class from `st.session_state["theme"]`), including how light/hybrid handle native widgets given the static
> config base. Keep `streamlit==1.50.0`, English UI, no engine/state changes, dev→main workflow. Verify each
> theme visually on every page before promoting.
