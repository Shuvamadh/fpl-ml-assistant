# FPL ML Assistant — Chart Audit & Visual Modernization Spec

Prepared against the code as of this writing:
`gui/charts.py`, `gui/chart_carousel.py`, `gui/main_window.py`, `gui/theme.py`,
`gui/llm_worker.py`.

All numbers quoted below were measured directly from `data/player_predictions.csv`
(652 rows × 32 cols) and `data/backtest_results.csv` (33 rows × 5 cols) at
GW2 of the 2025-26 season. Every palette claim was verified by running the
`dataviz` skill's `scripts/validate_palette.js` — the results are pasted inline,
not eyeballed.

The app exists to answer four decisions:

1. **Who do I captain?**
2. **Who do I start?**
3. **Who do I transfer in?**
4. **Am I beating my mini-league?**

A chart earns its place only if it changes one of those four answers. That is the
only test applied below.

---

## 1. Current inventory

Registered in `gui/main_window.py:318-336` into `ChartCarousel`
(`gui/chart_carousel.py`), 13 charts in 4 categories. All draw functions live in
`gui/charts.py`.

| # | Category | Carousel title | Draw function | Mark |
|---|---|---|---|---|
| 1 | Players | Starting XI predicted points | `draw_xi_points_bar` | horizontal bar, colored by position |
| 2 | Players | Value hunting (cost vs pts) | `draw_value_scatter` | scatter + top-12 annotated |
| 3 | Players | Top 5 player comparison (radar) | `draw_radar_comparison` | polar radar, 5 series |
| 4 | Players | Cost × Ownership × Pts (3D) | `draw_3d_value` | 3D scatter (mplot3d) |
| 5 | Players | Fixture difficulty heatmap | `draw_fixture_heatmap` | imshow, custom 3-hue cmap |
| 6 | Transfers | Squad value: bought vs now | `draw_squad_value_bars` | grouped vertical bars |
| 7 | Transfers | Best upgrades available | `draw_transfer_gains` | horizontal bar |
| 8 | Transfers | Price momentum vs ownership | `draw_price_momentum_scatter` | scatter, 3-color threshold |
| 9 | Mini League | League standings | `draw_league_standings_bar` | horizontal bar, me highlighted |
| 10 | Mini League | Most-owned players | `draw_ownership_chart` | horizontal bar, top 12 |
| 11 | Mini League | Captaincy choices | `draw_captaincy_pie` | **pie** |
| 12 | Mini League | Rank progression by GW | `draw_rank_progression` | multi-line, me emphasized |
| 13 | Model | Walk-forward validation (MAE) | `draw_backtest_mae` | 2-series line |

Shared styling: `_style_axes()` and `_accents()` in `gui/charts.py` read
`theme.current()` at draw time, so charts do follow the theme switcher. That
part is already right and should be preserved.

### Measured baseline of the data these charts render

`data/player_predictions.csv`, n = 652:

| Column | Key measurement |
|---|---|
| `status_ok` | 480 True / 172 False — **26.4% of rows are unavailable players** |
| `status` | a=480, u=99, i=57, d=15, s=1 |
| `position` | MID 289, DEF 214, FWD 78, GKP 71 |
| `pred_points_adj` | min 0.037, median 0.858, max 6.675, skew +1.09 |
| `minutes` | **44.2% are exactly 0**; only 34.5% have played ≥ 90 |
| `selected_by_percent` | median 0.2%, **skew +6.02**; 452/652 (69%) are below 1%; only 41 above 10%; max 72.3 (Haaland) |
| `now_cost_m` | only **38 distinct values**, 80% of available players at ≤ £5.5m, skew +3.25 |
| `form` / `total_points` / `ep_next` | **45.1% / 45.1% / 42.2% exact zeros** |
| `expected_goal_involvements` | **54.9% exact zeros**, skew +3.62 |
| `defensive_contribution` | **52.0% exact zeros** |
| `fdr_next_n_mean` | only **5 distinct values** (2.6, 2.8, 3.0, 3.2, 3.4) across all 20 teams |
| `next_fdr` | 4 distinct values: 2→165, 3→330, 4→89, 5→68 |
| `price_signal` | integer −5…+5, 24.5% zeros, roughly symmetric (skew +0.03) |
| `price_flag` | FALL 214, RISE 30, null 408 |
| `chance_of_playing_next_round` | non-null on only 230 rows; of those **157 are 0.0** |
| `news` | non-null on 172 rows |

Correlations with `pred_points_adj` (n=652):
`pred_points` .992 · `value_ratio` **.958** · `minutes` **.861** · `form` .787 ·
`total_points` .787 · `ep_next` .771 · `xGI` .708 · `defensive_contribution` .693 ·
`now_cost_m` .554 · `selected_by_percent` .543 · `price_signal` .190 ·
`fdr_next_n_mean` −.026 · `next_fdr` −.119.

`data/backtest_results.csv`, n = 33 (GW6–GW38), 7 folds
(`fold_trained_before_gw` ∈ {6, 11, 16, 21, 26, 31, 36}):

- mean `mae_model` 0.9559, mean `mae_naive` 1.0490
- mean edge (naive − model) **0.0931**, range **0.0234 → 0.1800**
- **model beats naive in 33 of 33 gameweeks (100%)**
- `mae_model` spans only 0.861 → 1.053 — a **0.192 total range**
- `n_players` per row varies 581 → 1074 (GW34 = 581, GW33 = 1074) — folds are
  not equally weighted

---

## 2. Palette validation (run, not assumed)

The `dataviz` skill requires computing colorblind-safety rather than judging it.
Results from `scripts/validate_palette.js`:

**The app's current chart color cycle** — `_accents()` in `gui/charts.py`
(`#00ff87, #e90052, #38bdf8, #facc15, #a78bfa, #fb7185`) against the FPL Purple
panel `#2b0730`:

```
[FAIL] Lightness band   outside band: #00ff87 L=0.876, #38bdf8 L=0.754,
                        #facc15 L=0.861, #a78bfa L=0.709, #fb7185 L=0.719
[PASS] Chroma floor     all 6 >= 0.1
[PASS] CVD separation   worst adjacent #fb7185↔#a78bfa ΔE 17.6 (protan)
[PASS] Normal-vision    worst adjacent ΔE 21.1
[PASS] Contrast         all 6 >= 3:1
```

**Honest read:** the current colors are *not* colorblind-unsafe — that check
passes comfortably. The single real failure is the **lightness band**: five of six
hues sit at OKLCH L 0.71–0.88, far above the 0.48–0.67 band for a dark surface.
That is exactly why the charts read as neon and glaring against the app chrome.
The fix is to *step the same hues down*, not to replace them.

The proposed replacement 4-slot categorical set for positions
(`#3987e5` blue, `#d95926` orange, `#199e70` aqua, `#c98500` yellow) against the
new surface `#171a26`:

```
[PASS] Lightness band   all 4 inside L 0.48-0.67
[PASS] Chroma floor     all 4 >= 0.1
[PASS] CVD separation   worst adjacent #c98500↔#199e70 ΔE 8.4 (protan)
[PASS] Normal-vision    worst adjacent ΔE 19.8
[PASS] Contrast         all 4 >= 3:1
→ ALL CHECKS PASS
```

For scatter charts (all-pairs comparison, a stricter gate) only the **first three
slots** validate; a 4th position hue on a scatter is not permitted:

```
#3987e5,#d95926,#199e70  --pairs all  →  ALL CHECKS PASS
  worst all-pairs CVD ΔE 9.4 (deutan), normal-vision ΔE 20.9
```

Consequence: **never color a scatter by all four positions.** Use position as
*shape* (composite encoding) or facet into small multiples.

Warning for anyone tempted to keep the brand pair as two data series after
dimming them: `#00a35c` (dimmed FPL green) vs `#e0407a` (dimmed magenta) **FAILS**
CVD separation at **ΔE 2.9 (deutan)**. The full-brightness pair passes (ΔE 26.6)
purely because of its lightness difference. So the FPL green/magenta pairing is
only safe at its current extreme lightness — which fails the other gate. Do not
use green-vs-magenta as the semantic pair for rise/fall. Use the validated
diverging pair below.

**Diverging pair** for price momentum and any above/below-baseline encoding
(`#3987e5` blue ↔ `#e0603c` orange) on `#171a26`:

```
[PASS] all checks — CVD ΔE 24.5 (protan), normal-vision ΔE 31.2
```

Blue↔orange is a genuine warm/cool opposition; the current
green/yellow/magenta ramp in `draw_fixture_heatmap` is a **rainbow**, which the
anti-pattern catalog forbids outright for a magnitude scale.

---

## 3. Chart verdicts

### KEEP — 4 charts

---

#### 1. Starting XI predicted points — `draw_xi_points_bar` — **KEEP, minor changes**

Directly answers *who do I start* and *who do I captain*. Horizontal bar for
magnitude with long text labels is the textbook-correct form. The 11 values are
sorted, which is the whole job.

*Measured justification for keeping it as bars:* among the top 10 predicted
players the gap between #1 and #2 is **0.397 pts** and the full #1→#10 spread is
**1.669 pts**. Those differences are small enough that only an aligned common
baseline resolves them — a bar chart is the only mark where a 0.4-unit difference
on a ~6-unit scale is reliably readable.

*Changes:*
- Position colors are currently four categorical hues, but position is **not the
  subject** of this chart — predicted points is. Switch to **emphasis encoding**:
  the captain pick in the accent hue, the other 10 in the de-emphasis gray.
  Keep position as a small text prefix in the tick label (`MID · Salah`), not a
  hue. This kills the 4-hue-for-a-1-number anti-pattern.
- Direct-label the bar ends with one decimal; drop the x-axis grid.
- 4px rounded ends on the free end of each bar, square at the baseline.

---

#### 2. Best transfer upgrades — `draw_transfer_gains` — **KEEP as-is (restyle only)**

Answers *who do I transfer in*, sorted magnitude, horizontal bars, `nlargest(10)`,
already de-duplicated on `in`. The empty-state message at
`gui/charts.py` in `draw_transfer_gains` is good practice and should be copied to
the other charts, which currently have no empty state.

*One change:* `pred_gain` can be negative for a downgrade-to-fund move; use the
**validated diverging pair** (blue for gain, orange for loss) with a zero rule,
rather than a single accent for everything.

---

#### 3. League points progression — `draw_rank_progression` — **KEEP**

The only chart in the app that answers *am I beating my mini-league* over time
rather than at a single instant, and it already uses correct **emphasis
encoding**: rivals in `t["border"]` gray at alpha 0.6, my line in the accent at
2.5px with markers. That is exactly the recommended treatment for
"one series is the point, the rest are context."

*Changes:* label my line at its right endpoint instead of relying on the legend
box; thin the rival lines to 1px solid hairline; add a subtle direct label on the
current-leader line only.

---

#### 4. Squad value: bought vs now — `draw_squad_value_bars` — **KEEP but change the form**

The decision it supports (what can I actually sell for) is real. But
before→after per item is a **dumbbell chart's** job, not grouped bars. Grouped
bars burn twice the horizontal space and force the eye to compare two bar heights
that are, for most players, within £0.1–0.3m of each other — `now_cost_m` has only
**38 distinct values across the entire 652-player set**, in £0.1m steps, so the
two grouped bars are usually near-identical in height and the difference (which
*is* the information) is left for the reader to compute.

*Change to:* horizontal **dumbbell** — one row per player, a gray dot at buy
price, an accent dot at current price, a connecting rule colored by direction
(blue = gained, orange = lost), sorted by profit. The current 60°-rotated
7pt x-tick labels also go away, which is a legibility win on its own.

---

### CUT — 3 charts

---

#### 5. Cost × Ownership × Predicted Points (3D) — `draw_3d_value` — **CUT. No hedging.**

This chart should be deleted. Four measured reasons:

1. **The third axis is redundant.** Among the 480 available players the three
   axes are mutually correlated: cost↔ownership **r = .573**, cost↔pred_points
   **r = .610**, ownership↔pred_points **r = .531**. All three axes are largely
   measuring "is this player good." A 3D scatter spends an entire extra spatial
   dimension — the most expensive channel available — on a variable that is
   already 53–61% explained by the other two.
2. **The data is a wall, not a cloud.** 80.2% of available players sit at
   `now_cost_m` ≤ £5.5m, and cost has only 34 distinct values among them (48
   players at exactly £4.0m, 109 at £4.5m, 113 at £5.0m). The x-axis is
   effectively a handful of discrete planes, so the "3D cloud" is really a few
   flat sheets viewed at an angle.
3. **Ownership is unplottable as a linear axis.** `selected_by_percent` has
   skew **+6.02**: 452 of 652 players are below 1% and one player (Haaland) is at
   72.3%. On a linear axis, 69% of the data compresses into the first 1.4% of the
   axis length. It is a single smear at the origin plus one dot far away.
4. **No occlusion control, no rotation, no tooltips.** `depthshade=True` on a
   static `FigureCanvasQTAgg` means near points hide far points with no way to
   interrogate them. There is no hit-testing, no label, no interaction — you
   cannot read a single player's name off it. A chart from which no individual
   value can be recovered supports no decision.

Nothing here is recoverable by tuning. The information it gestures at
(cheap + good + low-owned = a differential) is delivered better and exactly by
the proposed **Differential Finder** chart in §4. Delete `draw_3d_value`, drop
the `from mpl_toolkits.mplot3d import Axes3D` import and the `three_d` branch of
`_style_axes()`, and remove the carousel entry at `main_window.py:322`.

---

#### 6. Top 5 player comparison (radar) — `draw_radar_comparison` — **CUT.**

Also should be deleted, for reasons specific to this data rather than a general
dislike of radars:

1. **The axes are collinear, so the polygons are nearly similar shapes.** The
   four attributes it plots correlate as: `pred_points_adj`↔`value_ratio`
   **r = .947**, `pred_points_adj`↔`form` **r = .755**,
   `pred_points_adj`↔`xGI` **r = .703**, `value_ratio`↔`form` **r = .721**. A
   radar's only claim is that it reveals *shape differences* between profiles.
   When every axis moves with every other at r ≈ .70–.95, all five polygons are
   scaled copies of one another and the shape carries no signal.
2. **The normalization is computed over 5 rows, not the population.** The
   `(x - min) / (max - min)` in `draw_radar_comparison` is applied to the
   *selected* subframe. So whoever is 5th-best always renders at 0 on that axis
   and whoever is best always renders at 1.0. The chart says "Semenyo has zero
   xGI" when he actually has **0.71**, and the shape changes entirely if a
   different 5 players are picked. This is a correctness bug, not a taste issue.
3. **Area is a misleading channel.** Radar area scales with the square of the
   radius, so a player 20% better on every axis renders with ~44% more area.
4. **Axis order is arbitrary** — the attribute list is built by a
   `dict.fromkeys` over a hardcoded list, and rotating that order changes every
   polygon's apparent shape while changing no data.

Replace with a **small-multiples bar panel**: 4 mini horizontal-bar charts (one
per attribute), 5 players each, percentile-ranked **against the full 480-player
available population** rather than against each other. Same information, no area
distortion, no false zeros, and each attribute keeps a common baseline.

---

#### 7. Captaincy choices (pie) — `draw_captaincy_pie` — **CUT the pie, KEEP the question.**

This is the clearest "pie should be a bar" case in the app.

- `draw_captaincy_pie` sorts by `captained_by_n` and slices *every* distinct
  captain with **no `head()` cap**. In a typical mini-league of 10–20 managers
  that is commonly 6–12 distinct captains. Past ~6 segments adjacent slices blur
  and the `_accents()` list of 6 colors is **cycled** —
  `(_accents() * (len(df) // len(_accents()) + 1))[:len(df)]` — so a 7th captain
  is painted **exactly the same hue as the 1st**. Two different players, same
  color, in the same chart. That is an unconditional defect.
- Angle is the worst-read magnitude channel; the counts are small integers where
  the reader wants "4 vs 3," not "23% vs 17%."
- `autopct="%1.0f%%"` shows percentages of a small denominator, so with 12
  managers the labels read 8%, 8%, 8%, 8% — which discards the actual count.

*Replace with:* a horizontal bar of `captained_by_n`, sorted descending, capped at
the top 8 with the tail folded into a literal **"Other (n)"** row, my own captain
in the accent hue and everyone else gray (emphasis, since "did I pick the
template captain or a differential" is the actual decision), and the raw integer
count direct-labeled at each bar end. No percentages.

---

### CHANGE — 3 charts

---

#### 8. Value hunting (cost vs pts) — `draw_value_scatter` — **CHANGE**

The decision is sound but the encoding fights the measured distribution.

- **Severe overplotting on a quasi-discrete x-axis.** Cost takes only 34 distinct
  values among available players, with 109 players stacked at £4.5m and 113 at
  £5.0m. This is not a scatter, it is 34 vertical strips. And **41.9% of available
  players have `pred_points_adj` < 1.0**, so the bottom fifth of the plot is a
  solid bar of ink.
- *Fix:* filter to `minutes > 0` before plotting (removes **140 of 480** available
  players who have literally not played a minute and whose predictions are
  therefore priors, not evidence), and apply horizontal jitter of ±£0.03m to break
  the strips.
- **Add the Pareto frontier as a line** — the upper-left hull of the cloud. That
  is what "value hunting" actually means, and drawing it removes the need for the
  reader to eyeball it.
- **Reduce annotation.** 12 labels on a 480-point cloud is dense. Label only the
  frontier points. Note that top-`value_ratio` skews cheap-and-defensive: 6 of the
  current top 12 by `value_ratio` are ≤ £5.0m and 7 of 12 are DEF, so the labels
  clump in one corner and collide.
- **Do not color by position on this chart** — the all-pairs palette gate caps a
  scatter at 3 categorical hues (see §2). Use one hue plus shape-by-position, or
  facet into a 2×2 small-multiple grid by position, which also fixes the fact that
  `value_ratio` medians differ by position (DEF .219, MID .177, FWD .151,
  GKP **.013**) — GKPs are a completely different regime and pooling them is
  misleading.

---

#### 9. Fixture difficulty heatmap — `draw_fixture_heatmap` — **CHANGE (colormap is wrong)**

Right form — a heatmap is correct for a team × gameweek grid of an ordered
1–5 scale — wrong colors.

- The current cmap is `LinearSegmentedColormap.from_list("fdr", [accent2, "#facc15", accent])` =
  **green → yellow → magenta**. That is a rainbow on a magnitude scale, forbidden
  outright. Worse, it *looks* diverging (it has a distinct midpoint hue) while
  encoding a scale whose midpoint FDR=3 has no "neutral" meaning — though in FPL
  practice FDR 3 genuinely is neutral, so **diverging is the right family, just
  with the right hues.**
- *Fix:* use the validated diverging pair anchored at FDR = 3:
  `#3987e5` (blue, easy, FDR 2) → neutral gray `#6b7280` (FDR 3) → `#e0603c`
  (orange, hard, FDR 5). Validated: CVD ΔE 24.5, normal-vision ΔE 31.2, both
  ≥ 3:1 contrast.
- **Only 4 of the 5 FDR levels occur** in the current data (`next_fdr` ∈ {2,3,4,5};
  no FDR 1 anywhere), and 330 of 652 rows are FDR 3. Keep `vmin=1, vmax=5` so the
  scale is stable week to week, but the legend must be a **discrete 5-swatch
  legend, not a continuous colorbar** — the underlying variable is an integer with
  5 levels and `fig.colorbar()` misrepresents it as continuous.
- `ax.text(..., color=t["bg"])` writes cell values in the *background* color,
  which on a mid-ramp cell will be low-contrast. Compute per-cell text color from
  cell luminance instead.
- Sort rows by mean FDR ascending so the easiest run is at the top. Currently row
  order is whatever `fixture_df.index` happens to be.

---

#### 10. Price momentum vs ownership — `draw_price_momentum_scatter` — **CHANGE**

Real decision (when do I have to buy before a rise), but three encoding faults:

- **The y-variable is discrete.** `price_signal` takes only **11 integer values**
  (−5…+5) with the distribution −3→160, 0→160, 1→98, −1→78, −2→58, −5→43,
  3→25, −4→11, 2→14, 5→3, 4→2. Plotting 652 points on 11 discrete y-levels
  produces 11 horizontal lines of overlapping dots. This is a
  **strip / beeswarm** by `price_signal` level, or a **diverging bar of counts**,
  not a scatter.
- **The x-variable is unusable linearly.** `selected_by_percent` skew +6.02;
  69% below 1%. Use a **log or square-root x-scale**, or bucket ownership into
  ordered bands (<1%, 1–5%, 5–10%, 10–25%, >25%) and render a diverging stacked
  bar of price_signal within each band.
- **The color coding is upside-down and mislabeled.** The code assigns
  `t["accent2"]` (green) to `price_signal >= 3` and `t["accent"]` (magenta) to
  `<= -3`, and titles it "green=rising, pink=falling" — but only **30 players
  carry `price_flag == "RISE"` versus 214 `FALL`**, so 214 players get painted in
  the alarm color and the chart reads as a sea of magenta. Also, the annotation
  loop labels `df[price_signal >= 3].nsmallest(8, 'selected_by_percent')` — the 8
  *least-owned* risers — which is at most 8 of a 30-player pool but is not
  described anywhere in the chart. Use the validated blue↔orange diverging pair
  with a genuine neutral at 0, and state the label rule in the subtitle.

---

#### 11. Most-owned players in your league — `draw_ownership_chart` — **KEEP, one change**

Correct form (horizontal bar, sorted, top 12). The one change: this chart answers
"what is the template" but not "where am I different from it," which is the
decision. **Add a second mark**: a small filled/hollow indicator per row for
whether *I* own that player. That converts a descriptive chart into a
differential-audit chart at near-zero cost.

---

#### 12. League standings — `draw_league_standings_bar` — **KEEP, one change**

Correct emphasis encoding already (me in `accent2`, rivals in `accent`). But it
uses the **FPL green vs FPL magenta pair as the only distinguishing channel**,
and per §2 that pair is only CVD-safe at its current excessive lightness. Switch
the rivals to the de-emphasis gray (`#3f4657`) and keep only me in the accent —
that is emphasis encoding proper, and it removes the dependency on a
green/magenta hue discrimination entirely. Also direct-label my own bar with the
points gap to the leader, which is the number actually being sought.

---

#### 13. Walk-forward validation MAE — `draw_backtest_mae` — **CHANGE substantially**

This is the chart with the widest gap between what it plots and what it means.

*What the file actually contains:* 33 gameweeks, GW6–GW38, grouped into
**7 folds** by `fold_trained_before_gw` ∈ {6, 11, 16, 21, 26, 31, 36}. The
current `draw_backtest_mae` plots `mae_model` and `mae_naive` as two continuous
lines against `GW` and **completely ignores the fold column** — so it draws an
unbroken line across fold boundaries where the model was actually retrained. The
line implies temporal continuity that does not exist; at GW11, GW16, GW21, GW26,
GW31 and GW36 a *different model* takes over.

*Two specific measured problems with the two-line encoding:*

- The two lines never cross and never come close. `mae_model` spans
  0.861–1.053; `mae_naive` spans 0.959–1.171. The **model wins 33 out of 33
  gameweeks (100%)**. Two nearly-parallel lines separated by a roughly constant
  0.093 tell you nothing you can't read from one sentence — and the quantity the
  reader wants (*how much better?*) is the vertical gap, which the eye is bad at
  measuring between two non-parallel curves.
- The full `mae_model` range is only **0.192 units** on an axis matplotlib will
  auto-scale to roughly 0.85–1.18. Almost the entire visual variation in that
  chart is noise.

**Recommendation — replace with two charts:**

**13a. Model edge over baseline, by gameweek, faceted by fold.**
Encode the **difference** (`mae_naive − mae_model`), not two series. Justification
from the numbers: the difference is the entire question (does the model beat the
naive baseline, and by how much), it is strictly positive in all 33 rows so a
zero baseline is a meaningful reference the eye can use, and its range
(0.0234 → 0.1800, mean 0.0931) uses the full plot height instead of 15% of it.
Draw it as a **column chart with a zero rule**, colored with the diverging pair so
any future negative gameweek is immediately visible. Facet into **7 small
multiples, one per fold**, so fold boundaries are structural (panel breaks) rather
than invisible. Per-fold mean edge, direct-labeled on each panel:

| fold trained before GW | mean edge |
|---|---|
| 6 | 0.0767 |
| 11 | 0.0572 |
| 16 | **0.1312** |
| 21 | 0.0982 |
| 26 | 0.1048 |
| 31 | 0.0864 |
| 36 | 0.1000 |

That 2.3× spread between the weakest fold (0.0572) and the strongest (0.1312) is
genuinely interesting and is **entirely invisible** in the current chart.

**13b. Per-fold spread as a strip/box summary.**
7 folds × ~5 gameweeks each. A horizontal strip plot — one row per fold, one dot
per gameweek's edge, with the fold mean as a heavier rule — shows both level and
consistency. This directly answers "is the model reliably better or just better on
average," which two averaged lines cannot.

**Caveat to state on the chart:** `n_players` per gameweek ranges **581 to 1074**,
so MAE rows are not equally precise. Encode `n_players` as **dot size** in 13b, or
at minimum note the range in the subtitle. Averaging these rows unweighted (as any
"overall MAE" headline would) is not quite right.

**Also add a hero stat tile above these**, not a chart: *"Model beats baseline in
33/33 gameweeks · mean MAE 0.956 vs 1.049."* Per the form heuristic, a single
headline number belongs in a stat tile, not squeezed into a plot.

---

## 4. Missing charts — what the columns support but nothing renders

Every one of these is buildable from columns already present in
`data/player_predictions.csv`.

---

### M1. Minutes reliability / rotation risk — **highest priority missing view**

This is the single largest omission. `minutes` correlates with `pred_points_adj` at
**r = .861** — the strongest relationship in the file after the prediction's own
components — and **44.2% of all rows have minutes = 0**. Among the 480 *available*
players the distribution is:

| minutes (2 GWs played) | count |
|---|---|
| 0 | **140** |
| 1–45 | 95 |
| 46–135 | 79 |
| 136–179 | 72 |
| **180 (every minute)** | **94** |

So only **94 of 480 available players are actually nailed-on**, and 140 available
players have not played at all. Nothing in the app shows this. A player with a
high `pred_points_adj` and 40 minutes played is a completely different proposition
from one with the same prediction and 180 minutes — and that distinction decides
*who do I start*.

**Chart:** horizontal dot plot for the current squad + transfer shortlist,
x = minutes played as a share of available minutes, with the 3 risk bands shaded
(<50% rotation risk, 50–90% probable, >90% nailed). Emphasis coloring: my squad in
accent, shortlist in gray.

---

### M2. Availability triage panel

`status_ok` is False for **172 of 652 (26.4%)** rows and `news` is non-null on
exactly those 172. `chance_of_playing_next_round` is non-null on 230 rows, of
which **157 are 0.0** and only 15 are in the ambiguous 25/50/75 band.
Critically: **6 players who are NOT `status_ok` still have `pred_points_adj` > 2.0**
— i.e. the model rates them as startable while the API says they are not
available. There are also **15 players flagged `status == 'd'` (doubtful)**.

Nothing in the charts tab surfaces this, and it is a direct *who do I start*
failure mode. This is **not a chart — it is a status list** (per the form
heuristic: more than ~7 classes with meaning → a table). Render as a compact
alert strip: player, status glyph, chance-of-playing %, the `news` string, and the
predicted points being forfeited. Status colors only (reserved tokens), each with
an icon and text label, never color alone.

---

### M3. Differential finder (the honest replacement for the 3D chart)

The information the 3D scatter gestured at, done in 2D and readable:
x = `selected_by_percent` on a **log scale** (mandatory — skew +6.02), y =
`pred_points_adj`, filtered to `status_ok & minutes > 0`. Quadrant rules at the
league-relevant ownership threshold (~5%) and at a points threshold. The
upper-left quadrant *is* the differential list. Label only that quadrant.

Measured support: among the top 10 predicted players, ownership ranges from
**6.1% (Foden) to 72.3% (Haaland)** — a genuine 12× spread at essentially
equivalent predicted output (5.43 vs 6.67 pts). That is the differential decision,
and it is currently invisible.

---

### M4. Prediction uncertainty — **the most important structural gap**

`pred_points_adj` is rendered everywhere as a bare point estimate with no
interval. But the app has the material for an honest uncertainty band:
`backtest_results.csv` gives a measured **mean absolute error of 0.956 points**,
stable across 33 gameweeks (range 0.861–1.053).

That number is decisive for captaincy. The **#1 vs #2 captain gap is 0.397 points**
and the #1 vs #10 spread is **1.669 points** — both *smaller than or comparable
to* the model's own ±0.96 MAE. Stated plainly: **the model cannot distinguish the
top ~9 captaincy candidates from one another.** Presenting a ranked list with no
interval implies a precision the backtest says does not exist.

**Chart:** captaincy shortlist as a **dot plot with ±MAE error bars**, plus a
shaded band showing which candidates fall inside the leader's interval. This is
the single highest-value addition in this document — it changes the captain
decision from "pick the top row" to "these 9 are a coin flip, so break the tie on
fixture and minutes."

*Implementation note:* if per-player prediction quantiles ever become available
from the LightGBM model, use those instead of a flat MAE band. The flat band is
the honest interim, and must be labeled as such.

---

### M5. Fixture run vs prediction — currently near-useless as built

Flagged for a reason the data makes stark: `fdr_next_n_mean` has only **5 distinct
values across all 20 teams** (2.6, 2.8, 3.0, 3.2, 3.4 — a total range of 0.8 on a
1–5 scale, sd 0.235), and it correlates with `pred_points_adj` at **r = −0.026** —
effectively zero. `next_fdr` does slightly better at **r = −0.119**, still
negligible.

**This is worth saying out loud in the plan:** the fixture-difficulty signal, as
currently computed over the next-N horizon, has almost no discriminating power
between teams. Either the horizon N is too long (averaging washes out the
variation) or the FDR source is too coarse. The fixture heatmap (#5) should be
kept because *per-gameweek* FDR (`next_fdr`, 4 distinct levels, 165/330/89/68)
does vary usefully. But do **not** build a chart on `fdr_next_n_mean` as a
continuous predictor — with a 0.8-unit range across 20 teams it will produce a
flat, meaningless plot. Flag this back to whoever owns `src/fixtures_fdr.py`.

---

### M6. Price momentum as a ranked watchlist, not a scatter

`price_signal` is an 11-level integer and only **30 players are flagged RISE vs
214 FALL**. The decision ("must I buy this player tonight") concerns roughly 30
players, not 652. Replace the scatter's job with a **ranked list of the top ~15
risers and top ~15 fallers**, sorted by `price_change_percent` (which is
continuous, 399 distinct values, range −130 to +134.4), with my own squad members
marked. That is a lollipop/diverging bar, and it is directly actionable in a way a
652-point scatter is not.

---

## 5. Summary table

| Chart | Verdict | Key measured reason |
|---|---|---|
| XI predicted points | **Keep**, emphasis recolor | top-10 spread only 1.669 pts — needs common baseline |
| Value hunting scatter | **Change** | cost has 34 distinct values; 41.9% of pts < 1.0 |
| Radar comparison | **CUT** | axes correlate r = .70–.95; normalization over 5 rows creates false zeros |
| 3D cost×own×pts | **CUT** | axes correlate .53–.61; 80.2% of players ≤ £5.5m; ownership skew +6.02; no interaction |
| Fixture heatmap | **Change** colormap | green→yellow→magenta rainbow on an ordered scale |
| Squad value bars | **Change** to dumbbell | prices in £0.1m steps, 38 distinct values — grouped bars hide the delta |
| Transfer gains | **Keep** | correct form already |
| Price momentum scatter | **Change** | y has 11 discrete levels; x skew +6.02; 214 FALL vs 30 RISE |
| League standings | **Keep**, gray the rivals | green/magenta pair only CVD-safe at failing lightness |
| Most-owned | **Keep** + "do I own" mark | descriptive → differential at no cost |
| Captaincy pie | **CUT the pie** → bar | uncapped slices, `_accents()` cycles: 7th captain = same hue as 1st |
| Rank progression | **Keep** | already correct emphasis encoding |
| Backtest MAE | **Change** → delta + per-fold facets | fold column ignored; model wins 33/33; mae range only 0.192 |
| **+ Minutes reliability** | **ADD** | minutes r = .861; only 94/480 available players are nailed-on |
| **+ Availability triage** | **ADD** (table) | 172 unavailable; 6 of them predicted > 2.0 pts |
| **+ Differential finder** | **ADD** | top-10 ownership spans 6.1%–72.3% at similar predicted pts |
| **+ Prediction uncertainty** | **ADD** | MAE 0.956 > the 0.397 gap between captain #1 and #2 |
| **+ Price watchlist** | **ADD** | only 30 RISE flags — a list, not a scatter |

Net: 13 charts → 10 charts + 1 table + 1 stat tile. Three cut outright, five
substantially re-encoded, five added.

### Proposed carousel categories after the change

`gui/main_window.py:318-336`, `ChartCarousel.add_chart` calls, reorganized around
the four decisions rather than around data sources:

- **Captain & Start** — XI predicted points (emphasis), Captaincy uncertainty
  (M4), Minutes reliability (M1), Availability triage (M2)
- **Transfers** — Best upgrades, Value frontier, Differential finder (M3),
  Price watchlist (M6), Squad value dumbbell
- **Mini League** — Standings (emphasis), Points progression, Captaincy bar,
  Template vs me
- **Model** — Edge-over-baseline by fold, Per-fold spread, hero stat tile
- **Fixtures** — FDR heatmap (diverging, discrete legend)

---

# 6. Visual modernization spec

Target: another engineer implements this without design decisions of their own.
Every value is literal.

## 6.1 Color palette — `gui/theme.py`, `THEMES` dict

Add a **new** theme entry named `"Graphite"` and make it the default in
`gui/config.py`'s fallback. Do not delete the existing five themes — the
switcher in `main_window.py:172-176` depends on them and users have a stored
preference in `data/gui_config.json` (currently `"FPL Purple"`).

The existing `THEMES` dict has 9 keys (`bg, bg2, bg3, panel, accent, accent2,
text, text_dim, border`). Adding keys is a breaking change for the other themes
unless every theme gets them. **Add the new keys to all five existing themes with
sensible values**, or gate them with `.get()` in `stylesheet()`.

```python
"Graphite": {
    # --- surfaces (existing keys) ---
    "bg":       "#0f1116",   # window / app ground
    "bg2":      "#171a26",   # chart surface, table header, elevated rail
    "bg3":      "#1e2230",   # hover / alternate row / pressed
    "panel":    "#171a26",   # cards, chart canvas  (== bg2 deliberately)
    # --- ink ---
    "text":     "#e8eaf0",   # primary
    "text_dim": "#9aa3b5",   # secondary / axis labels
    "border":   "#272c3a",   # hairlines, 1px only
    # --- brand accents ---
    "accent":   "#e0407a",   # FPL magenta, dimmed — CHROME ONLY, not data
    "accent2":  "#00d977",   # FPL green, dimmed  — CHROME ONLY, not data
    # --- NEW keys ---
    "text_mute":  "#6b7488",  # tertiary, disabled, placeholder
    "focus":      "#3987e5",  # focus ring — one hue, everywhere
    "series_1":   "#3987e5",  # blue    (validated, dark)
    "series_2":   "#d95926",  # orange  (validated, dark)
    "series_3":   "#199e70",  # aqua    (validated, dark)
    "series_4":   "#c98500",  # yellow  (validated, dark) — adjacent forms only
    "series_mute":"#3f4657",  # de-emphasis gray for emphasis encoding
    "div_pos":    "#3987e5",  # diverging positive pole
    "div_neg":    "#e0603c",  # diverging negative pole
    "div_mid":    "#6b7280",  # diverging neutral midpoint
    "good":       "#199e70",
    "warn":       "#c98500",
    "bad":        "#d95926",
    "critical":   "#e34948",
},
```

**Hard rule, and the most important line in this spec:**
`accent` and `accent2` are **chrome colors** — buttons, the selected tab
underline, the progress bar. They are **not data colors**. Chart series come from
`series_1..4`, `series_mute` and the diverging trio. This is what stops the
neon-on-purple look and it is what `_accents()` in `gui/charts.py` currently
violates by returning `[accent2, accent, ...]` as its cycle.

Validator results for `series_1..4` on surface `#171a26` are in §2: all five
checks PASS. Scatter/all-pairs forms cap at `series_1..3`.

## 6.2 Spacing scale — 4px base

Use only these values. `gui/main_window.py` currently mixes 16, 12, 14, 10, 8, 6,
4 semi-arbitrarily.

| Token | px | Use |
|---|---|---|
| `s1` | 4 | icon gaps, inside-chip padding |
| `s2` | 8 | control vertical padding, label→field |
| `s3` | 12 | between controls in a row |
| `s4` | 16 | card inner padding, tab content padding |
| `s5` | 24 | between major sections |
| `s6` | 32 | window margins on wide layouts |

Applications:
- `main_window.py` `_build_ui`, `layout.setContentsMargins(16,16,16,16)` → keep
  (= s4); `layout.setSpacing(12)` → **24** (s5), the header/cards/tabs stack is
  currently cramped.
- `stat_card()` `layout.setContentsMargins(14,10,14,10)` → **`(16,12,16,12)`**.
- `chart_carousel.py` `_build_ui`, `layout.setSpacing(8)` → **12** (s3).
- Every tab's `QVBoxLayout` should get `setContentsMargins(16,16,16,16)` — they
  currently get Qt defaults (9px), which is why tab content sits tight to the pane
  border.

## 6.3 Type scale — pt, Segoe UI Variable → Segoe UI → Inter fallback

| Role | Size | Weight | Color | Where |
|---|---|---|---|---|
| Display / hero stat | 28pt | 600 | `text` | new stat-card value |
| `QLabel#Header` | 17pt | 700 | `text` | `main_window.py:149`, `:258` |
| Section title | 13pt | 600 | `text` | "Starting XI", "Bench" labels |
| Body / table cell | 10pt | 400 | `text` | default `QWidget` rule |
| Control label | 10pt | 500 | `text_dim` | header row labels |
| `QLabel#SubHeader` / caption | 9pt | 400 | `text_dim` | `main_window.py:151` |
| Stat-card label | 8.5pt | 600, `letter-spacing: 0.6px`, uppercase | `text_mute` | `stat_card()` |
| Tabular numerals | 10pt | 400 | `text` | all numeric table columns |

Two notes:
- The base rule in `theme.stylesheet()` is `font-size: 13px` and `Header` is
  `22px`. Switch the whole sheet to **pt** so it respects Windows DPI scaling.
  13px ≈ 10pt; 22px ≈ 17pt.
- **`letter-spacing` is not supported in Qt stylesheets.** It must be applied via
  `QFont.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)` on the label widget in
  `stat_card()`. Flagged below in §6.11.

## 6.4 Radii & elevation

| Element | Radius |
|---|---|
| Cards (`QFrame#StatCard`), chart canvas frame, tab pane | 10px |
| Buttons, inputs, combos | 8px |
| Table container | 10px; **rows: 0** |
| Chips / badges / status pills | 999px (pill) |
| Progress bar + chunk | 4px |

Qt has no box-shadow in stylesheets. Elevation is expressed as **surface
lightness step only**: `bg` → `bg2`/`panel` → `bg3`, each with a `1px solid
border` hairline. Do not attempt shadows via stylesheet.

## 6.5 Tables — `gui/table_utils.py` (`fill_table`) and every `QTableWidget`

Current state (`theme.stylesheet()`): `alternate-background-color: bg3`,
`gridline-color: border`, `QTableWidget::item { padding: 4px; }`.

Target:

- **Row height 34px** (`table.verticalHeader().setDefaultSectionSize(34)`).
  Currently unset, so rows are ~22px and the app reads as a spreadsheet.
- **Kill the vertical grid, keep a horizontal hairline.** Full-grid tables are the
  single most dated element in this app. Qt cannot draw only horizontal
  gridlines via stylesheet, so:
  `table.setShowGrid(False)` in code, then in the sheet
  `QTableWidget::item { border-bottom: 1px solid #1c2130; padding: 0 12px; }`.
  (`#1c2130` = a `border` one step darker than the row, so it recedes.)
- **Drop zebra striping**: `table.setAlternatingRowColors(False)`. With a
  34px row and a bottom hairline, zebra is redundant noise. Remove
  `alternate-background-color` from the sheet.
- **Header**: `background: #171a26; color: #9aa3b5; font-size: 9pt; font-weight: 600;
  padding: 10px 12px; border: none; border-bottom: 1px solid #272c3a;`
  and `table.horizontalHeader().setHighlightSections(False)` (Qt bolds the header
  section of the selected column by default — dated, and it causes column reflow).
- **Hover**: `QTableWidget::item:hover { background: #1e2230; }`. Requires
  `table.setMouseTracking(True)`, otherwise hover only fires while a button is
  held.
- **Selected**: `background: rgba(57,135,229,0.18); color: #e8eaf0;` — a tinted
  fill, not the current opaque `selection-background-color: accent` (solid
  magenta behind text is unreadable at 10pt).
  Also `table.setSelectionBehavior(QAbstractItemView.SelectRows)`.
- **Numeric columns right-aligned with tabular figures.** `fill_table` in
  `table_utils.py` should set `Qt.AlignRight | Qt.AlignVCenter` on any column
  whose formatter is numeric.
- `table.setFrameShape(QFrame.NoFrame)` and put the 10px radius + 1px border on a
  wrapping `QFrame` — **Qt does not clip `QTableWidget` content to a
  border-radius**, so a radius set directly on the table produces square corners
  over a rounded border. Flagged in §6.11.
- Note `fill_table(..., row_color_col="price_flag")` at `main_window.py:555`
  tints whole rows. Retint from `good`/`bad` at **12% alpha**, not a saturated
  fill, and add a text glyph (▲ / ▼) in the flag column so it is not color-alone.

## 6.6 Buttons — `theme.stylesheet()`

```
QPushButton {
    background: #e0407a; color: #ffffff; border: none;
    border-radius: 8px; padding: 9px 18px;
    font-size: 10pt; font-weight: 600;
}
QPushButton:hover   { background: #ea5a8c; }
QPushButton:pressed { background: #c22f64; padding-top: 10px; padding-bottom: 8px; }
QPushButton:disabled{ background: #272c3a; color: #6b7488; }
QPushButton:focus   { outline: none; border: 2px solid #3987e5; padding: 7px 16px; }
```

**Fix a real bug:** the current sheet has
`QPushButton:hover { background-color: {accent}; opacity: 0.9; }` —
`opacity` is **not a supported Qt stylesheet property**, and the background is
the same color as the base state, so the hover rule is a complete no-op. The app
currently has no button hover feedback at all. Replace with an explicit lighter
hex as above.

`QPushButton#Secondary` keeps the transparent/bordered treatment; add
`:pressed { background: #272c3a; }`.

## 6.7 Inputs, combos, tabs, scrollbars

```
QLineEdit, QComboBox {
    background: #0f1116; border: 1px solid #272c3a; border-radius: 8px;
    padding: 8px 12px; color: #e8eaf0; font-size: 10pt;
    selection-background-color: #3987e5;
}
QLineEdit:hover, QComboBox:hover { border-color: #3f4657; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #3987e5; background: #171a26; }
QLineEdit::placeholder { color: #6b7488; }   /* see §6.11 */
QLineEdit:disabled { background: #14161d; color: #6b7488; border-color: #1e2230; }
```

Note the field background is **darker** than the panel (`#0f1116` on `#171a26`) —
inset inputs on a raised surface. That is the current-generation dark-UI
convention and the app currently does the opposite (`panel` on `bg`).

The fixed widths at `main_window.py:159/164/169` (100/50/100px) should become
`setMinimumWidth` with the same values; at 10pt with 12px padding a 5-digit team
ID will clip at a fixed 100px.

Tabs — keep the underline pattern, tighten it:
```
QTabWidget::pane { border: 1px solid #272c3a; border-radius: 10px; top: -1px;
                   background: #171a26; }
QTabBar::tab { background: transparent; color: #9aa3b5;
               padding: 11px 16px; margin-right: 2px;
               font-size: 10pt; font-weight: 600;
               border-bottom: 2px solid transparent; }
QTabBar::tab:hover    { color: #e8eaf0; background: #171a26;
                        border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { color: #e8eaf0; border-bottom: 2px solid #00d977; }
```
(3px → 2px underline; 18px → 16px horizontal padding. With 8 tabs — My Squad,
Squad Value, Transfers, All Players, Price Watch, Mini League, Charts, AI
Assistant — the bar is at risk of scrolling at 1280px window width. Set
`self.tabs.setUsesScrollButtons(True)` and
`self.tabs.setElideMode(Qt.ElideNone)` explicitly rather than leaving it to the
platform default.)

Scrollbars:
```
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #272c3a; border-radius: 5px; min-height: 32px; }
QScrollBar::handle:vertical:hover { background: #3f4657; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
```
The `add-line`/`sub-line`/`add-page` rules are **required** — without them Qt
renders default arrow buttons and a gray trough that ignore the handle styling.
The current sheet omits them, which is why the scrollbars still look native.

Progress bar (`main_window.py:190-193`, indeterminate `setRange(0,0)`):
```
QProgressBar { background: #171a26; border: none; border-radius: 4px;
               height: 4px; text-align: center; color: #9aa3b5; font-size: 9pt; }
QProgressBar::chunk { background: #00d977; border-radius: 4px; }
```
Set `self.progress.setTextVisible(False)` and `setFixedHeight(4)` — an
indeterminate bar with a percentage label is meaningless, and a 4px hairline bar
reads as modern where a 20px bordered box reads as Windows XP.

## 6.8 Stat cards — `stat_card()` at `main_window.py:107-121`

```
QFrame#StatCard { background: #171a26; border: 1px solid #272c3a; border-radius: 10px; }
QFrame#StatCard:hover { border-color: #3f4657; }
QLabel[class="StatCardValue"] { font-size: 22pt; font-weight: 600; color: #e8eaf0; }
```
The label under the value gets 8.5pt/600/`#6b7488`, uppercase (already uppercased
in code via `label.upper()`).

Two caveats:
- `QLabel[class="StatCardValue"]` works only because the code calls
  `val.setProperty("class", "StatCardValue")`. If the property is set **after**
  the stylesheet is applied, Qt does not re-polish. `_restyle_stat_cards()` at
  `main_window.py:191` presumably handles this; whatever it does must call
  `widget.style().unpolish(w); widget.style().polish(w)` after any property change.
- The "Suggested Captain" card holds a player name, not a number. Give that one
  card 14pt instead of 22pt or long names will clip — 5 cards in a
  `QHBoxLayout` at 1280px is ~250px each.

## 6.9 Matplotlib rcParams — new module `gui/mpl_style.py`

The charts currently inherit matplotlib's defaults for everything the code does
not explicitly set: DejaVu Sans at 10pt, 0.8pt spines, default 6.4×4.8 figure
proportions, `tight_layout` with default pads. That is the main reason the plots
read as a foreign object dropped into the app chrome.

Create a **new file** (permitted — `gui/mpl_style.py` does not exist) exporting:

```python
def apply(t: dict) -> None:
    """t = theme.current(). Call at the top of every draw_* function,
    before _prep(), so a theme switch takes effect on redraw."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":   t["panel"],
        "figure.edgecolor":   "none",
        "figure.dpi":         110,          # was 100 — crisper on 1080p+
        "savefig.facecolor":  t["panel"],
        "axes.facecolor":     t["panel"],
        "axes.edgecolor":     t["border"],
        "axes.linewidth":     0.8,
        "axes.labelcolor":    t["text_dim"],
        "axes.labelsize":     9,
        "axes.labelpad":      8,
        "axes.titlecolor":    t["text"],
        "axes.titlesize":     11,
        "axes.titleweight":   "600",
        "axes.titlelocation": "left",       # left-aligned titles, like the UI
        "axes.titlepad":      14,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,        # value axis carries the grid instead
        "axes.spines.bottom": True,
        "axes.grid":          True,
        "axes.axisbelow":     True,         # grid behind the marks, always
        "grid.color":         t["border"],
        "grid.linewidth":     0.7,
        "grid.alpha":         0.6,
        "grid.linestyle":     "-",          # never dashed
        "xtick.color":        t["text_dim"],
        "ytick.color":        t["text_dim"],
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "xtick.major.size":   0,            # no tick marks; labels suffice
        "ytick.major.size":   0,
        "xtick.major.pad":    6,
        "ytick.major.pad":    6,
        "text.color":         t["text"],
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Segoe UI Variable Text", "Segoe UI",
                               "Inter", "DejaVu Sans"],
        "font.size":          9,
        "legend.frameon":     False,        # the box is chartjunk
        "legend.fontsize":    8.5,
        "legend.labelcolor":  t["text"],
        "lines.linewidth":    2.0,
        "lines.markersize":   5,
        "lines.solid_capstyle": "round",
        "patch.linewidth":    0,
        "scatter.edgecolors": "none",
        "figure.autolayout":  False,        # use constrained_layout instead
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
    })
```

And a series accessor to replace `_accents()`:

```python
def series(t, n=1):
    """Categorical slots in FIXED order. Never cycle: n > 4 must fold to
    'Other' or facet. Scatter / all-pairs forms cap at n = 3 (see GUI_PLAN §2)."""
    slots = [t["series_1"], t["series_2"], t["series_3"], t["series_4"]]
    if n > 4:
        raise ValueError("fold the tail into 'Other' or facet; do not cycle hues")
    return slots[:n]
```

Notes on adopting this:
- `_style_axes()` in `gui/charts.py` becomes almost empty — most of its body is
  superseded. Keep it as a thin shim so the 13 call sites do not all need editing.
- Every `fig.tight_layout()` call must be **removed** when
  `constrained_layout` is enabled; the two conflict and matplotlib emits a
  warning and then ignores one of them.
- `axes.grid: True` plus `axes.spines.left: False` means the grid line at each
  y-tick replaces the left spine. On horizontal bar charts, override to
  `ax.grid(axis="x")` only — a grid parallel to the bars is noise.
- Set the y-axis grid off entirely on any chart with fewer than 6 categories.
- `ChartCanvas.__init__` (`gui/charts.py`) hardcodes `dpi=100` and
  `figsize=(6,4.5)`. Change to `dpi=110`; keep the figsize but raise
  `setMinimumHeight(320)` → **380**, because with the new 14px title pad and 6px
  tick pad the plot area at 320px gets squeezed.

## 6.10 AI Assistant tab — `main_window.py:340-356`, `:557-592`

This is the widget where default Qt styling looks worst, and it has three
distinct problems: appearance, the pending state, and how the assistant's output
is rendered.

### 6.10.1 Rendering approach — decide this first

`self.chat_log` is a read-only `QTextEdit` and the code already appends **HTML**
(`f"<b>You:</b> {question}"`, `<i>Thinking...</i>`, an inline
`<span style='color:#e90052'>`). So rich text is already in play.

**Verdict: switch `QTextEdit` → `QTextBrowser`, and render each message as an
HTML block.** Reasons:

- `QTextBrowser` is a `QTextEdit` subclass, so `append()`, `textCursor()`,
  `setReadOnly()` and the existing `_remove_thinking_line()` logic all still work
  — it is a near drop-in.
- It is read-only by default (no risk of the log becoming editable), it handles
  link activation, and it gives `setOpenExternalLinks(True)` for free, which
  matters the moment the model emits a URL.
- The current `QTextEdit` will happily let a user place a caret in the log; a
  chat transcript should not have an I-beam caret.

**Bubbles are achievable, but not via stylesheet.** Qt's stylesheet system styles
the *widget*, not the rich-text document inside it. A per-message rounded bubble
must be built as HTML inside the document. Qt's rich text engine supports a
useful-but-limited HTML/CSS subset:

- **Supported and sufficient for a good bubble:** `<table>` with
  `bgcolor`/`background-color`, `cellpadding`, `width`, `align`; `<div>` with
  `background-color`, `margin`, `padding`; `<p style="line-height:150%">`;
  `<b>`, `<i>`, `<code>`, `<pre>`, `<ul>`, `<ol>`, `font-family`, `color`.
- **NOT supported:** `border-radius` (so a true rounded bubble is impossible),
  `box-shadow`, flexbox, `max-width` as a percentage of the viewport,
  `text-align: justify`.

**Therefore: do not specify rounded bubbles.** Specify **speaker-prefixed blocks
with a left accent rule** — which Qt's subset renders correctly and which looks
deliberate rather than like a failed bubble:

```python
USER_BLOCK = (
    '<table width="100%" cellpadding="10" cellspacing="0" '
    'style="margin:6px 0;"><tr>'
    '<td style="background-color:#1e2230; border-left:3px solid #3987e5;">'
    '<div style="color:#9aa3b5; font-size:9pt; font-weight:600;">You</div>'
    '<div style="color:#e8eaf0; line-height:150%;">{body}</div>'
    '</td></tr></table>'
)
ASSISTANT_BLOCK = (
    '<table width="100%" cellpadding="10" cellspacing="0" '
    'style="margin:6px 0;"><tr>'
    '<td style="background-color:#171a26; border-left:3px solid #00d977;">'
    '<div style="color:#9aa3b5; font-size:9pt; font-weight:600;">Assistant</div>'
    '<div style="color:#e8eaf0; line-height:150%;">{body}</div>'
    '</td></tr></table>'
)
```

Sender differentiation is therefore carried by **three** channels — the label
text, the left rule color, and the block background — never by color alone.

**Proportional, not monospace,** for prose: 10pt Segoe UI at `line-height:150%`.
Monospace (`Cascadia Mono, Consolas, monospace`, 9.5pt) applies **only** inside
`<code>`/`<pre>` spans, on a `#0f1116` ground.

### 6.10.2 The assistant's markdown-ish output

`llm_assist.ask()` returns a plain string from a local Qwen3 model, and
`_on_chat_answer` currently drops it straight into an HTML `append()`. Two
consequences:

1. **This is an HTML-injection path.** Any `<` in the model's output — and a
   model discussing formations or comparisons will emit them — corrupts the
   document. The user's own question is interpolated raw too.
2. Markdown the model emits (`**bold**`, `- bullets`, ` ```code``` `) renders as
   literal asterisks and backticks.

**Required:** convert markdown → HTML before insertion, and escape everything
that is not produced by that conversion. Order matters:
`html.escape(raw)` first, then apply a small markdown transform over the escaped
text (`**x**`→`<b>`, `` `x` ``→`<code>`, `^- ` →`<li>`, ` ``` ` blocks →`<pre>`,
`\n\n`→`</p><p>`). Do **not** pull in a markdown library and then trust its
output — Qt's parser is lenient and will render injected tags. The user's
question at `main_window.py:564` needs `html.escape()` regardless of anything
else in this spec.

### 6.10.3 Widget styling

```
QTextBrowser#ChatLog {
    background: #0f1116;
    border: 1px solid #272c3a;
    border-radius: 10px;
    padding: 12px;
    font-size: 10pt;
    color: #e8eaf0;
    selection-background-color: #3987e5;
}
QLineEdit#ChatInput {
    background: #171a26; border: 1px solid #272c3a; border-radius: 8px;
    padding: 11px 14px; font-size: 10pt; color: #e8eaf0;
}
QLineEdit#ChatInput:focus    { border: 1px solid #3987e5; background: #1e2230; }
QLineEdit#ChatInput:disabled { background: #14161d; color: #6b7488;
                               border-color: #1e2230; }

QPushButton#ChatSend {
    background: #00d977; color: #0f1116; border: none; border-radius: 8px;
    padding: 11px 22px; font-size: 10pt; font-weight: 700; min-width: 84px;
}
QPushButton#ChatSend:hover    { background: #1ae88a; }
QPushButton#ChatSend:pressed  { background: #00b862; }
QPushButton#ChatSend:disabled { background: #272c3a; color: #6b7488; }
```

Dark text (`#0f1116`) on the green send button is deliberate — `#00d977` is far
too light for white text to clear 4.5:1.

Set the object names in `_build_ui`:
`self.chat_log.setObjectName("ChatLog")`,
`self.chat_input.setObjectName("ChatInput")`,
`self.chat_send_btn.setObjectName("ChatSend")`. Without them these rules leak to
every `QLineEdit` and `QPushButton` in the app.

Layout: `chat_input_row` gets `setSpacing(12)` and
`ai_layout.setContentsMargins(16,16,16,16)`; the descriptive `QLabel` at
`main_window.py:342` gets `setWordWrap(True)` — at 1280px it currently forces a
horizontal minimum on the whole tab.

### 6.10.4 The pending / "thinking" state — required, currently inadequate

`_send_chat` appends a static `<i>Thinking...</i>` and disables both controls. A
local Qwen3 model on CPU can take 20–60+ seconds, during which the only feedback
is three static words and two grayed-out widgets. That reads as a hang.

Specify all five of the following:

1. **An animated ellipsis in the log.** A `QTimer` at 400ms cycling
   `Thinking · Thinking·· Thinking···`, rewriting the last block in place (the
   existing `_remove_thinking_line()` already demonstrates the cursor technique).
   Motion is what distinguishes "working" from "hung."
2. **An elapsed-second counter** after ~3s: `Thinking··· (12s)`. For a call with
   no progress signal, elapsed time is the only honest progress indicator and it
   sets expectations on the second use.
3. **An indeterminate `QProgressBar`** (`setRange(0,0)`), 3px tall, no text,
   placed directly beneath the chat log, `setVisible(True)` for the duration.
   Reuse the styling from §6.7. This is the strongest "not frozen" signal
   available and costs four lines.
4. **The send button changes label and role**: text `"Ask"` → `"Thinking…"`,
   disabled. Better still, make it a **Stop** button — but note `AskWorker.run()`
   in `gui/llm_worker.py` has no cancellation point (it blocks inside
   `llm_assist.ask`), so a true cancel needs cooperation from `src/llm_assist.py`.
   If that is out of scope, keep it disabled and say so; do not ship a Stop button
   that does not stop.
5. **Auto-scroll to the bottom** on every append —
   `self.chat_log.verticalScrollBar().setValue(maximum())`. `QTextEdit.append()`
   does **not** guarantee this once the user has scrolled up, so a long answer can
   arrive entirely off-screen and look like nothing happened.

Also: on failure, `_on_chat_error` writes a hardcoded `#e90052` at
`main_window.py:590`. That bypasses the theme entirely and will be invisible or
garish under some of the five themes. Use `t["critical"]` (`#e34948`) from the
active theme, in an error block with a left rule in the same color plus the word
"Error" — not color alone.

**Robustness note (not styling, but adjacent and worth flagging):**
`ask_llm()` in `gui/llm_worker.py` connects `worker.answered` and `worker.failed`
to `worker.deleteLater` — but only `answered` is connected
(`worker.answered.connect(worker.deleteLater)`; there is no matching line for
`failed`). On the error path the worker is never deleted. Whoever owns that file
should add `worker.failed.connect(worker.deleteLater)`.

## 6.11 Qt limitations — cannot be done with stylesheets alone

| Wanted | Why not | Do this instead |
|---|---|---|
| Drop shadows / elevation | No `box-shadow` in QSS | Surface lightness step + 1px hairline (§6.4), or `QGraphicsDropShadowEffect` on the widget in code (expensive; do not put it on table rows) |
| `letter-spacing` on the stat-card label | Not a QSS property | `QFont.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)` in `stat_card()` |
| `opacity` on hover | Not a QSS property — **the current sheet's `QPushButton:hover` rule is a silent no-op** | Explicit lighter hex per state (§6.6) |
| Rounded corners on `QTableWidget` | Qt does not clip the viewport to a `border-radius` | Wrap in a `QFrame` that carries the radius + border; `table.setFrameShape(QFrame.NoFrame)` |
| Horizontal-only gridlines | `gridline-color` is all-or-nothing | `setShowGrid(False)` + `QTableWidget::item { border-bottom: 1px solid … }` |
| `border-radius` on chat bubbles | Qt rich text does not support it (nor `box-shadow`, flexbox, `%` `max-width`) | Left accent rule + block background via `<table>` (§6.10.1) |
| `::placeholder` pseudo-element | Not supported in QSS in all Qt versions | `QPalette.PlaceholderText` role, set on the widget's palette |
| Transitions / animations | QSS has no `transition` | `QPropertyAnimation`, or accept instant state changes (recommended — instant is fine here) |
| Per-row alternating tint by value | `alternate-background-color` is positional only | Set `QBrush` per item in `fill_table` (`table_utils.py`) — already the pattern used by `row_color_col` |
| Styling the `QTabBar` scroll buttons | `QTabBar::scroller` support is patchy across styles | Prevent overflow: shorter tab labels, or move Charts/AI to a second row |
| Native title-bar theming | Windows owns the frame | Leave it, or `DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE)` via `ctypes` on the `winId()` — one call, worth doing, the light title bar over a dark app is conspicuous |

## 6.12 Application order

1. `gui/theme.py` — add the `Graphite` theme and the new keys (add them to all
   five existing themes too, or `.get()`-guard them in `stylesheet()`).
2. `gui/theme.py` — rewrite `stylesheet()` per §6.3–6.8 and §6.10.3. Biggest
   single visual win, zero risk to logic.
3. New file `gui/mpl_style.py` (§6.9); call `apply(theme.current())` at the top
   of each `draw_*` in `gui/charts.py`; delete every `fig.tight_layout()`.
4. `gui/table_utils.py` + each `QTableWidget` construction — row height, no grid,
   no zebra, row selection, mouse tracking, right-aligned numerics (§6.5).
5. `gui/main_window.py` — spacing tokens, object names on the chat widgets,
   `setWordWrap`, progress-bar sizing, stat-card font sizes.
6. `gui/charts.py` — delete `draw_3d_value` and `draw_radar_comparison`; convert
   `draw_captaincy_pie` to a bar; recolor everything off `series_*` rather than
   `_accents()`.
7. AI tab: `QTextEdit` → `QTextBrowser`, `html.escape` + markdown transform, the
   five pending-state elements (§6.10.4).
8. New charts M1–M6, in the priority order M4 (uncertainty) → M1 (minutes) →
   M2 (availability) → M3 (differentials) → M6 (price) → 13a/13b (backtest).
