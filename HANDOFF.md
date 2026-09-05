# Handoff — FPL ML Assistant

Living status doc for whoever (human or Claude session) picks this project up next.
Update this file whenever you land a meaningful change — treat it as the single
source of truth for "what's the state right now," since this project has had
multiple parallel Claude sessions working on it (ML/`src/` in one, GUI/`gui/`
in another) and cross-session messages don't leave a permanent record here.

## Links

- **GitHub**: https://github.com/Shuvamadh/fpl-ml-assistant (public)
- **Live Streamlit app**: https://fpl-ml-assistant-ap5qfe6ydjfawg7v4jkyvr.streamlit.app/
  (the *current* one — an earlier deploy at a different `.streamlit.app` slug got
  permanently stuck on old code and was deleted; if this one ever needs the same
  treatment, delete + recreate via share.streamlit.io rather than debugging reboots)
- **User's team**: 8041052 "Neon Knights"
- **Mini-league**: 1766517 "Khasi Leauge" (24 managers, user is admin)

## What's working right now (as of this writing)

- **ML pipeline** (`src/`): LightGBM points model, trained opponent-strength +
  real fixture-venue features, honest decision-quality metrics, walk-forward +
  rolling-origin validation, a P(minutes≥60) classifier exposed as
  `start_probability` (kept separate from the main prediction — the naive
  combination was worse, see README). Chip strategy optimizer (`chips.py`,
  ILP via `pulp`), mini-league projections + banter stats (`league_projection.py`).
- **Desktop GUI** (`gui/`, PySide6): full GUI_PLAN.md visual modernization applied
  (Graphite theme, redone charts, chip/league tabs wired in, AI Assistant tab with
  markdown rendering). Smoke-tested clean end to end on 2026-09-05 — full refresh
  through every stage (predictions, squad, chips, league standings/squads/rank
  progression/projections/fun-stats/effective-ownership) with no errors.
  **Desktop shortcut installed** on this machine via `install_shortcut.ps1` →
  `C:\Users\Shuvam\Desktop\FPL Assistant.lnk` (pythonw.exe, no console window,
  custom icon from `assets/fpl.ico`). Re-run that script any time the project
  moves or Python is reinstalled.
- **Streamlit app** (`streamlit_app/app.py`): feature parity with desktop minus
  AI Assistant (only appears if Ollama is reachable at startup — never true on
  Streamlit Cloud, so it's cleanly absent there, not broken). Includes a
  "Viewing as" picker so any of the 24 mini-league managers can see the app
  built around their own squad. Reuses `gui/league_extras.py` directly (it's
  deliberately Qt-free) but does NOT import anything else from `gui/`, since
  the rest of that package pulls in PySide6, which this deployment doesn't
  install (`streamlit_app/requirements.txt` is deliberately scoped down).
- **Automated retraining**: `.github/workflows/retrain.yml`, weekly (Mondays
  06:00 UTC) + manual trigger, reruns the full pipeline and pushes an updated
  model if anything changed — which also triggers Streamlit Cloud's
  auto-redeploy. Needs the `workflow` OAuth scope on whoever's `gh` token
  pushes changes to it (see "Gotchas" below).
- **Everything is committed to the repo** — no more gitignored data files.
  User explicitly asked for this after hitting missing-file crashes on a
  fresh deploy three times in a row (see Gotchas). `data/history_gws.csv` is
  ~54MB; GitHub warns (recommended max 50MB) but doesn't block (100MB hard
  limit) — leave it, don't move to Git LFS unless it actually becomes a
  problem.
- **Streamlit-specific analysis views added** (2026-09-05, per the desktop
  GUI_PLAN.md chart audit's highest-priority findings, adapted for
  Streamlit): captain shortlist with ±MAE error bars (flags when top
  candidates are statistically indistinguishable — the audit's top finding),
  rotation-risk table off `start_probability` (not raw minutes), squad- and
  pool-wide availability triage (status flags the model doesn't always
  know), a log-scale differential finder (2D replacement for the desktop's
  cut 3D scatter), and Price Watch reframed as ranked risers/fallers lists.
  All matplotlib-based (`streamlit_app/requirements.txt` now includes it),
  transparent backgrounds since per-viewer theme isn't server-detectable.
  Verified rendering correctly on the live Streamlit Cloud deploy, not just
  locally.

## Known gaps / possibly worth doing next

- No real desktop *installer* (PyInstaller/MSI) — the shortcut launches the
  raw Python script via `pythonw.exe`. Fine for the user's own machine;
  would need packaging work to hand to someone without this dev environment.
- Two-stage hurdle model's regressor half (`models/points_regressor_started.txt`)
  is trained but unused in production — a smarter combination than a flat
  non-starter baseline might beat the single-stage model; not attempted.
- `collect_history.py`'s `SEASONS` list doesn't include the current live
  season (2026-27) since it's incomplete in the source archive — the trained
  model only learns from fully-completed seasons by design (see README
  "Model" section for why this is intentional, not an oversight). The live
  *predictions* already incorporate the current season's form via `predict.py`'s
  own direct API calls, independent of when the model was last trained.
- No git tags/releases yet — everything's just commits on `main`.

## Gotchas hit (don't rediscover these)

- **Windows git corrupts LightGBM model files** via autocrlf line-ending
  conversion — fixed with `.gitattributes` marking `models/*.txt` and
  `*.parquet` as `binary`. If a new binary-ish text format ever gets added
  to the repo, add it there too, don't assume.
- **Streamlit Cloud can get permanently wedged** on stale code even after
  multiple pushes and manual "Reboot app" clicks — if a deploy is stuck
  showing an error that's provably already fixed in the repo, stop
  debugging and delete+recreate the app instead.
- **`gh` push to `.github/workflows/*.yml` needs the `workflow` OAuth
  scope**, which a normal `repo`-scoped token doesn't have — GitHub rejects
  the push with `refusing to allow an OAuth App to create or update workflow`.
  Fix: `gh auth refresh -h github.com -s workflow` (interactive device-code
  flow — needs the human to approve in a browser, can't be done unattended).
- **`get_predictions()`-style functions that return a DataFrame aren't
  enough** — `recommend.py`/`chips.py`/`league_projection.py` all read
  predictions back from `data/player_predictions.csv` on disk (matching the
  desktop GUI's `data_bridge.py` contract), so any new caller (like the
  Streamlit app) must actually write that file, not just hold the DataFrame
  in memory. This one caused a "works on my machine" bug that only showed
  up on a genuinely fresh deploy with no pre-existing file.
- **Any file write needs its own directory-exists check** — don't assume a
  fresh clone/deploy has the same directory tree a long-lived local dev
  copy accumulated (`data/raw/` not existing was the first deploy crash).
- Full list of subtler ones (pandas `.itertuples()` breaking on a column
  named `in`, `self.event` shadowing `QObject.event()`, etc.) is in
  README.md's "Gotchas hit while building this" section — check there too.

## Cross-session coordination notes

This project has been built across multiple parallel Claude sessions: one
on `src/`/`data/`/`models/` (ML), one on `gui/` (desktop UI/UX, via a
detailed `GUI_PLAN.md` spec), and this handoff doc itself created because
cross-session chat messages don't persist anywhere durable. **If you're a
new session picking this up: read this file first, then check `git log`
for what's actually landed (cross-session status messages can be stale by
the time you see them — verify against the repo, not just what a peer
claimed).**

---
*Last updated: 2026-09-05, after shifting focus to the Streamlit app —
added the chart-audit-derived analysis views and verified them on the live
Streamlit Cloud deployment, not just locally.*
