# FPL ML Assistant

Data-driven Fantasy Premier League decision support, with a desktop GUI: pulls
the official public FPL API + a multi-season historical archive, trains a
LightGBM model to predict each player's points for the upcoming gameweek, and
turns that into starting-XI, captaincy, transfer, and squad-value
recommendations for team **8041052** (Neon Knights) — plus a mini-league
tracker for **Khasi Leauge** (league ID 1766517).

## Desktop app

```bash
python gui/app.py
```

PySide6 app. Tabs: **My Squad** (optimal XI + bench, fixture-aware),
**Squad Value** (buy price -> real sell price, FPL's half-profit-on-rise
rule applied), **Transfers** (upgrade suggestions within real sellable
budget), **All Players** (searchable/filterable, ~650 players, full stat
profile), **Price Watch** (FPL's own rise/fall momentum signal), **Mini
League** (standings, ownership/template team, captaincy split, your
differentials, chips played, rank progression), **Charts** (paginated
carousel — XI points, value scatter, radar comparison, 3D cost/ownership/
points, fixture heatmap, squad value, transfer gains, price momentum,
league standings/ownership/captaincy/rank progression, walk-forward
backtest MAE), **AI Assistant** (local Ollama/Qwen3 chat grounded in your
live refreshed data — ask about captaincy, transfers, differentials in
plain language). 5 selectable themes.

Remembers your team ID / event / league ID / theme between launches
(`data/gui_config.json`). Refresh runs the whole pipeline on a background
thread so the UI doesn't freeze during the ~1min live data pull.

## Pipeline (src/)

```
fpl_api.py        official FPL API wrapper (bootstrap, fixtures, entry/picks, transfers, per-player history), file-cached
collect_history.py  pulls 10 seasons (2016-17..2025-26) of gameweek data -> data/history_gws.csv
features.py        leak-free rolling/lag feature engineering incl. opponent-strength -> data/features.parquet
train_model.py     trains LightGBM regressor + reports honest decision-quality metrics -> models/points_model.txt
                    also trains a P(minutes>=60) classifier (train_two_stage()) -> models/minutes_classifier.txt
backtest.py        walk-forward validation + rolling-origin cross-season CV -> data/backtest_results.csv
predict.py         scores all ~650 current players via live per-player API data -> data/player_predictions.csv
                    real next-fixture venue + opponent form (trained-in) + a small residual FDR nudge + FPL's price-change signal
                    + start_probability (rotation-risk column from the minutes classifier)
squad_value.py      reconstructs real purchase price (from public transfer history + GW1 price) and FPL's true sell price
fixtures_fdr.py     next-N-fixture difficulty per team, fixture difficulty matrix for the heatmap chart
recommend.py        squad-specific: optimal XI, captain/vice, transfer suggestions (uses real sell price + fixture-adjusted points)
mini_league.py       league standings, every rival's squad, ownership/template/captaincy/differentials/chips/rank-progression insights
league_projection.py  projected GW winner, per-manager transfer suggestions, banter stats (bench luck, value growth, template adherence)
chips.py             wildcard/free-hit squad optimizer (ILP via pulp), blank/double GW detection, triple captain / bench boost advice
llm_assist.py        local Ollama (qwen3:8b) wrapper -- grounds chat answers in the app's live refreshed data
```

### `chips.py` — return schemas for GUI consumption

- `detect_blank_double_gameweeks(n=5)` -> DataFrame: `team` (id), `event`, `n_fixtures`, `is_blank`, `is_double`.
- `wildcard_squad(team_id, current_squad, horizon=5)` / `free_hit_squad(team_id, current_squad)` -> dict:
  `status` ("Optimal"/solver status), `squad` (15-row DataFrame), `xi` (11-row, sorted by score desc),
  `bench` (4-row), `captain` (web_name), `captain_pred`, `total_xi_points`, `total_squad_cost`, `budget`,
  `budget_source` (string explaining what the budget number means), `caveat` (string, always present --
  read it before rendering, it explains the approximation for that specific chip).
- `triple_captain_candidates(squad_with_preds, blank_double=None)` -> DataFrame sorted by
  `tc_gain_vs_normal_captain` desc: `web_name`, `position`, `name`, `pred_points_adj`, `is_dgw`,
  `effective_pred`, `tc_gain_vs_normal_captain`.
- `bench_boost_value(squad_with_preds, bench_element_ids)` -> dict: `bench_boost_value` (float),
  `bench` (list of dicts), `risky_bench_players` (list of names to flag in the UI).
- `chips_used(team_id)` -> list of chip names already played (grey these out in a chip picker).

Every optimizer function returns the full candidate detail (not just a winner) specifically so the UI
can show the spread rather than implying false precision — these are point estimates off a model with
backtested MAE ~0.95-1.0, not certainties.

### `league_projection.py` — return schemas for GUI consumption

- `project_gw_winner(league_squads, standings, picks_from_gw)` -> DataFrame sorted by `projected_total`
  desc: `entry`, `entry_name`, `player_name`, `current_total`, `projected_gw_points`, `projected_total`,
  `current_rank`, `projected_rank`, `rank_change`, `captain_name`, `picks_from_gw`, `is_stale` (always
  True — render this as a visible disclaimer, not just a data flag; see module docstring for why).
- `per_manager_transfer_suggestions(league_squads, assumed_bank=0.0, top_n=2)` -> DataFrame:
  `entry`, `entry_name`, `out`, `out_pos`, `out_pred`, `in`, `in_club`, `in_cost`, `in_pred`,
  `pred_gain`, `assumed_bank` (surface this last column — it's a stated assumption, not their real bank).
- `banter_stats(league_squads, standings, insights)` -> dict:
  `template_adherence` (DataFrame: `entry`, `entry_name`, `template_players_owned`, `template_size`),
  `per_manager_stats` (DataFrame: `entry`, `entry_name`, `total_bench_points_lost`,
  `avg_bench_points_per_gw`, `squad_value_gw1`, `squad_value_now`, `squad_value_growth`,
  `gw_rank_swing` — this last one is the mini-league's own week-over-week rank movement, NOT FPL's
  global overall rank, which was an early bug here since it swings by millions and is meaningless for
  league banter), `known_gap` (string — season-long best/worst captain pick is deliberately not
  computed, too expensive; see docstring).

## Weekly workflow (CLI, if not using the GUI)

```bash
python src/predict.py
python src/recommend.py 8041052 <latest_finished_or_current_event>
python src/mini_league.py 1766517 <event> 8041052
```

Re-run `collect_history.py` + `features.py` + `train_model.py` only
occasionally (e.g. once a month, or after a full season completes) — the
model doesn't need retraining every gameweek, only the live predictions do.

## Model

LightGBM regression predicting a player's FPL points in a gameweek from
features available *before* that gameweek: rolling 3/5-GW averages of
points/minutes/ICT/xGI/BPS, season-to-date form, cost, home/away (the
**real** next-fixture venue, looked up from the fixtures endpoint — not a
placeholder), and **both teams'** recent scoring/conceding rate (own team
*and* opponent, recovered via a self-join on each season's fixtures rather
than an external team-id table — see `features.build_team_id_map`).
Training data: 2020-21 through 2024-25 (2025-26 held out); earlier seasons
(2016-17..2019-20) exist in the raw archive but lack position/team/xG
tracking, so they're excluded from training as mostly-uninformative rows,
not just for memory reasons — same MAE either way confirms it.

**Validated three ways, not one:**
- Season holdout: MAE **0.964** vs 1.059 naive baseline.
- Walk-forward (expanding-window retrain through 2025-26, predicting 5-GW
  blocks ahead each time, never looking forward): overall MAE **0.954**,
  and — the important part — accuracy *improves* as the season progresses
  (0.966 early-season -> 0.910 late-season), the correct signature of a
  model actually using accumulating in-season signal rather than overfitting
  to history.
- Rolling-origin cross-season CV (train on strictly-earlier seasons only,
  validate on each of the last 4 seasons in turn): checks whether the
  season-holdout number is stable or a lucky draw for 2025-26 specifically.
  See `data/rolling_origin_results.csv` for the per-season spread.

**The honest metrics, not just MAE**: raw MAE is dominated by the ~55-60%
of rows that are non-playing (0 minutes, 0 points) and trivially easy to
get right. Segmented by actual minutes played: 0-min MAE ~0.33, 1-59min MAE
~1.17, 60+min (started) MAE ~2.35 — the real difficulty, and the number
that matters for lineup decisions, is the last one. Decision-quality
metrics reported alongside: within-gameweek Spearman rank correlation
(~0.72, how well the model *orders* players, which is what picking/
captaining actually needs), captain regret vs the perfect pick, captain
lift over the pool average, and unconstrained top-11-by-model vs
top-11-by-points-per-game actual points gained. Run `python
src/train_model.py` to see current numbers.

Minutes/starts still dominate feature importance — rotation risk is the
single biggest driver of FPL point variance, more than underlying quality
stats. `predict.py` applies one remaining small post-hoc adjustment
(`FDR_ADJUSTMENT`, 1.08 easiest down to 0.92 hardest) on top of the model's
own (now trained-in) opponent-strength signal, as a residual nudge rather
than the primary fixture-difficulty mechanism it used to be.

**Honest finding, flagged by a peer review and confirmed by a controlled
check — don't overclaim what the opponent/is_home fixes bought**: aggregate
MAE barely moved (0.963 before -> 0.964 after adding trained opponent
features and fixing the is_home bug — statistically indistinguishable, not
a real improvement by that metric). Two things are true at once here: (1)
fixing `is_home=1` and training real opponent-strength features instead of
a hand-tuned multiplier were still the right calls — a known systematic bug
is worth fixing regardless of whether it moves the headline number, and a
principled trained signal beats an arbitrary one even at parity; (2) FPL's
own fixture-difficulty rating (FDR) turns out to have close to **zero**
measured correlation with predicted points, even controlling for position
(checked per-position, not just pooled — pooled correlation is confounded
by player quality dominating the variance, which is why the *right* test is
per-position, and even that shows |r| < 0.13 across GKP/DEF/MID/FWD). This
matches a known critique in the wider FPL analytics community: FDR is a
coarse team-strength proxy that ignores injuries, tactical matchups, and
individual role, and its marginal predictive power for a single gameweek is
genuinely weak once a player's own form/quality is accounted for — this
isn't a bug in `fixtures_fdr.py`, it's what a proper check actually shows.
Consequence: the small `FDR_ADJUSTMENT` nudge is deliberately kept *small*
(not widened) precisely because the data doesn't support a strong fixture
effect existing to justify a bigger one — inflating it would be fitting the
adjustment to a narrative rather than the evidence. `fdr_next_n_mean`
(the 5-fixture average used by `chips.py`'s wildcard horizon score) is
additionally compressed by simple averaging of a mostly-"3" categorical
scale (expected statistical shrinkage, not itself a separate bug) and is
documented in `chips.py` as a mild directional signal, not a strong one.

**Two-stage (hurdle) model tried and honestly evaluated, not adopted for the
main prediction**: `train_model.train_two_stage()` trains a separate P(minutes
>=60) classifier and an E[points | started] regressor, motivated by "minutes/
starts dominate feature importance" suggesting the single model is mostly a
will-he-play classifier wearing a regressor's clothes. Results: the
classifier is genuinely excellent (**AUC 0.950**) and the regressor alone
beats a same-rows-only naive baseline on started rows (MAE 2.337 vs 2.502).
But the naive combination (`p*E[points|started] + (1-p)*flat_baseline`) has
**worse** overall MAE than the single joint model (1.018 vs 0.964) — a flat
constant for the "didn't start" case is cruder than what the single model
already learns implicitly by conditioning continuously on form/minutes
features. Conclusion: kept the single-stage model as the actual point
predictor (`pred_points`/`pred_points_adj`, unchanged), but the classifier's
P(started) is genuinely useful on its own and is now exposed as a separate
`start_probability` column in `data/player_predictions.csv` — a real
rotation-risk indicator the single-stage model's point estimate doesn't
surface on its own. Models saved to `models/minutes_classifier.txt`,
`models/points_regressor_started.txt` (the latter currently unused in
production, kept for reference/future work — e.g. a smarter combination
than a flat baseline might still beat the single-stage model; not attempted
here since the evidence for the naive version didn't support switching).

## Known limitations

- The small residual `FDR_ADJUSTMENT` multiplier is heuristic, not learned.
- Price-change alerts use FPL's own `price_change_projections` likelihood
  field, which is the platform's own signal but not a guarantee.
- Early-season predictions (GW1-3) are inherently noisier — 1-2 games of
  current-season data is a small sample, so the model leans more on season
  cost/position priors until form accumulates.
- Mini-league squads are only visible for gameweeks whose deadline has
  passed (FPL doesn't expose other managers' *upcoming* picks, only yours).
- The two-stage hurdle model was tried and its naive combination underperformed
  the single-stage model (see "Model" section) — its classifier half is used
  for `start_probability`, but a smarter combination than a flat baseline for
  non-starters remains unexplored.

## Gotchas hit while building this (for future reference)

- The FPL API returns most player stat fields (form, xG, xA, ICT, etc.) as
  **strings**, not numbers — always `pd.to_numeric(..., errors="coerce")`.
- Never name a `QObject` subclass attribute `self.event` — it silently
  shadows `QObject.event()` and produces a cryptic `'int' object is not
  callable` deep inside `moveToThread`.
- Never call `sys.stdout.reconfigure()` at module import time in code that
  might get imported from a background `QThread` — if another thread is
  concurrently writing to a redirected stdout, it can deadlock silently
  (no exception, no error, just a hang). Keep it CLI-entrypoint-only.
- `pandas.itertuples()` breaks on a column literally named `in` (Python
  keyword) — the transfers dataframe has one; use `.iterrows()` +
  `row['in']` instead.
- On a memory-constrained machine, reading the full feature table can hit a
  genuine `pandas.errors.ParserError: ... out of memory` even with
  `usecols` trimming. Parquet (`features.parquet`, float32 rolling columns)
  fixed this — smaller on disk, typed, and reads a column subset without
  the C parser tokenizing the whole file first. Also: `del` intermediate
  DataFrames as soon as they're consumed into a `lgb.Dataset`, rather than
  holding df + train_df + val_df + both Datasets alive simultaneously.
- `train_model.py` asserts no training row has a season newer than
  `VAL_SEASON` — a guard against silently training on the future once
  2026-27 data lands in `history_gws.csv` and someone forgets to bump it.
