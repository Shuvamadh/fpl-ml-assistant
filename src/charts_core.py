"""Chart implementations shared by BOTH frontends.

Every function here takes a plain matplotlib Axes and a palette dict, and
draws into it. Nothing in this module imports Qt, Streamlit, or any backend,
so it is safe to import from anywhere.

Why this module exists
----------------------
These charts used to live only in gui/charts.py, where every function's first
argument was a ChartCanvas (a FigureCanvasQTAgg subclass). That welded them to
Qt, so streamlit_app/app.py could not import them and two of them were
hand-rewritten there instead -- while the other eleven were simply missing
from the web app. Two frontends, two implementations, one of them a strict
subset: any fix to a chart had to be made twice or it silently drifted.

The split is now:
    src/charts_core.py   the drawing logic, backend-agnostic  (this file)
    gui/charts.py        thin Qt wrappers: hand it canvas.figure's ax
    streamlit_app/app.py calls the same functions, passes the fig to st.pyplot

Adding a chart means adding it here once. Both apps get it.

Palette keys expected: bg, panel, text, text_dim, border, accent, accent2,
good, bad, series_1, series_mute. DEFAULT_PALETTE below is a theme-neutral
fallback that stays legible on both light and dark backgrounds, which is what
the Streamlit side needs (Streamlit's theme is per-viewer and not reliably
detectable server-side). The desktop side passes theme.current() instead.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Theme-neutral fallback. Mid-grays read acceptably against both a white and a
# near-black page background, so a server-rendered chart doesn't have to know
# which theme the viewer picked.
DEFAULT_PALETTE = {
    "bg": "#ffffff",
    "panel": "none",
    "text": "#888888",
    "text_dim": "#888888",
    "border": "#888888",
    "accent": "#ff4b4b",
    "accent2": "#00c853",
    "good": "#00c853",
    "bad": "#ff4b4b",
    "series_1": "#38bdf8",
    "series_mute": "#9ca3af",
}

POSITION_COLORS = {"GKP": "#00c853", "DEF": "#ff4b4b", "MID": "#38bdf8", "FWD": "#facc15"}

SERIES = ["#38bdf8", "#facc15", "#00c853", "#ff4b4b", "#a78bfa", "#fb923c"]


def _p(palette):
    """Merge a caller palette over the defaults so a partial theme dict (or
    None) never raises KeyError mid-draw."""
    if not palette:
        return dict(DEFAULT_PALETTE)
    merged = dict(DEFAULT_PALETTE)
    merged.update({k: v for k, v in palette.items() if v})
    return merged


def style_axes(ax, palette=None, three_d=False, transparent=False):
    """Shared axis chrome. transparent=True keeps the figure background clear,
    which is what the web app wants so charts sit on the page background
    rather than punching a colored rectangle into it."""
    t = _p(palette)
    if transparent:
        ax.patch.set_alpha(0)
        ax.figure.patch.set_alpha(0)
    else:
        ax.set_facecolor(t["panel"])
        ax.figure.set_facecolor(t["panel"])
    ax.tick_params(colors=t["text_dim"], labelsize=8)
    if not three_d:
        for spine in ax.spines.values():
            spine.set_color(t["border"])
            spine.set_alpha(0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=t["border"], linewidth=0.5, alpha=0.35)
    else:
        for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane_axis.pane.set_alpha(0)
        ax.grid(color=t["border"], linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_color(t["text_dim"])
    ax.yaxis.label.set_color(t["text_dim"])
    ax.title.set_color(t["text"])


def _empty(ax, msg, palette=None):
    t = _p(palette)
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=t["text_dim"],
            transform=ax.transAxes, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


# ---------------------------------------------------------------- players ---

def xi_points_bar(ax, xi: pd.DataFrame, palette=None, transparent=False):
    t = _p(palette)
    if xi is None or xi.empty:
        return _empty(ax, "No squad loaded", t)
    df = xi.sort_values("pred_points_adj")
    colors = [POSITION_COLORS.get(p, t["accent2"]) for p in df["position"]]
    ax.barh(df["web_name"], df["pred_points_adj"], color=colors)
    ax.set_xlabel("Predicted points (fixture-adjusted)")
    ax.set_title("Starting XI - predicted points")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


def captain_uncertainty(ax, candidates: pd.DataFrame, mae: float, palette=None,
                        transparent=False, top_n: int = 9):
    """Predictions drawn as point estimates imply a precision the backtest
    doesn't support. If the #1 and #2 gap is smaller than the model's own MAE
    they are a coin flip, so draw the error bar and shade the band."""
    t = _p(palette)
    if candidates is None or candidates.empty:
        return _empty(ax, "No candidates", t)
    df = candidates.head(top_n).sort_values("pred_points_adj")
    y = range(len(df))
    ax.errorbar(df["pred_points_adj"], y, xerr=mae, fmt="o", color=t["accent"],
                ecolor=t["text_dim"], elinewidth=1.5, capsize=3, markersize=6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["web_name"])
    ax.set_xlabel(f"Predicted points (+/-{mae:.2f} MAE)")
    leader = df["pred_points_adj"].max()
    ax.axvspan(leader - mae, leader + mae, color=t["accent"], alpha=0.08)
    ax.set_title("Captain shortlist")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.25)
    ax.grid(axis="y", visible=False)
    return ax


def value_scatter(ax, predictions: pd.DataFrame, palette=None, transparent=False):
    t = _p(palette)
    if predictions is None or predictions.empty:
        return _empty(ax, "No predictions", t)
    df = predictions[predictions["status_ok"]].dropna(subset=["now_cost_m", "pred_points_adj"])
    if df.empty:
        return _empty(ax, "No available players", t)
    ax.scatter(df["now_cost_m"], df["pred_points_adj"], s=14, alpha=0.4, color=t["text_dim"])
    top = df.sort_values("value_ratio", ascending=False).head(12)
    ax.scatter(top["now_cost_m"], top["pred_points_adj"], s=40, color=t["accent2"], zorder=3)
    for _, row in top.iterrows():
        ax.annotate(row["web_name"], (row["now_cost_m"], row["pred_points_adj"]),
                    color=t["text"], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cost (GBP m)")
    ax.set_ylabel("Predicted points")
    ax.set_title("Value hunting: cost vs predicted points")
    style_axes(ax, t, transparent=transparent)
    return ax


def value_by_position_box(ax, predictions: pd.DataFrame, palette=None, transparent=False):
    """Points-per-million distribution split by position -- 'which position
    is actually good value right now' isn't answerable from the value_scatter
    (cost vs points) or the raw player table alone; a boxplot per position
    makes the spread and the outliers visible at a glance."""
    t = _p(palette)
    if predictions is None or predictions.empty:
        return _empty(ax, "No predictions", t)
    df = predictions[predictions["status_ok"]].dropna(subset=["value_ratio", "position"])
    if df.empty:
        return _empty(ax, "No available players", t)
    order = [p for p in ["GKP", "DEF", "MID", "FWD"] if p in df["position"].unique()]
    groups = [df.loc[df["position"] == p, "value_ratio"].values for p in order]
    bp = ax.boxplot(groups, tick_labels=order, patch_artist=True, showfliers=True,
                     medianprops={"color": t["text"], "linewidth": 1.5},
                     flierprops={"markersize": 3, "markeredgecolor": t["text_dim"], "alpha": 0.5})
    for patch, pos in zip(bp["boxes"], order):
        color = POSITION_COLORS.get(pos, t["accent"])
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(t["text_dim"])
    ax.set_ylabel("Points per GBPm")
    ax.set_title("Value by position")
    style_axes(ax, t, transparent=transparent)
    return ax


def differential_scatter(ax, predictions: pd.DataFrame, own_thresh: float = 5.0,
                         palette=None, transparent=False):
    """Low-owned players with above-median predicted points."""
    t = _p(palette)
    if predictions is None or predictions.empty:
        return _empty(ax, "No predictions", t)
    pool = predictions[(predictions["status_ok"]) & (predictions["minutes"] > 0)].copy()
    pool["selected_by_percent"] = pd.to_numeric(pool["selected_by_percent"], errors="coerce")
    pool = pool.dropna(subset=["selected_by_percent", "pred_points_adj"])
    pool = pool[pool["selected_by_percent"] > 0]
    if pool.empty:
        return _empty(ax, "No ownership data", t)
    pts_thresh = pool["pred_points_adj"].median()
    ax.scatter(pool["selected_by_percent"], pool["pred_points_adj"], s=14,
               alpha=0.35, color=t["text_dim"])
    diffs = pool[(pool["selected_by_percent"] < own_thresh) & (pool["pred_points_adj"] > pts_thresh)]
    ax.scatter(diffs["selected_by_percent"], diffs["pred_points_adj"], s=40,
               color=t["accent"], zorder=3)
    for _, row in diffs.nlargest(10, "pred_points_adj").iterrows():
        ax.annotate(row["web_name"], (row["selected_by_percent"], row["pred_points_adj"]),
                    color=t["text"], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.axvline(own_thresh, color=t["border"], linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(pts_thresh, color=t["border"], linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Owned % (log scale)")
    ax.set_ylabel("Predicted points")
    ax.set_title(f"Highlighted: under {own_thresh:.0f}% owned, above-median points")
    style_axes(ax, t, transparent=transparent)
    return ax


def radar_comparison(ax, predictions: pd.DataFrame, players=None, palette=None):
    """Normalised spider chart. ax MUST be created with projection='polar'."""
    t = _p(palette)
    df = predictions[predictions["status_ok"]].copy()
    if players:
        df = df[df["web_name"].isin(players)]
    else:
        df = df.sort_values("pred_points_adj", ascending=False).head(5)
    if df.empty:
        return _empty(ax, "No players selected", t)

    attrs = ["pred_points_adj", "expected_goal_involvements", "ict_index", "value_ratio", "form"]
    attrs = [a for a in dict.fromkeys(attrs) if a in df.columns]
    labels = {"pred_points_adj": "Pred Pts", "expected_goal_involvements": "xGI",
              "ict_index": "ICT", "value_ratio": "Value", "form": "Form"}

    norm = df[attrs].apply(pd.to_numeric, errors="coerce").fillna(0)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(attrs), endpoint=False).tolist()
    angles += angles[:1]

    for i, (_, row) in enumerate(df.iterrows()):
        values = norm.iloc[i].tolist()
        values += values[:1]
        c = SERIES[i % len(SERIES)]
        ax.plot(angles, values, color=c, linewidth=2, label=row["web_name"])
        ax.fill(angles, values, color=c, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([labels.get(a, a) for a in attrs], color=t["text"], fontsize=8)
    ax.set_yticklabels([])
    ax.patch.set_alpha(0)
    ax.spines["polar"].set_color(t["border"])
    ax.grid(color=t["border"], alpha=0.4)
    ax.set_title("Player comparison (normalised)", color=t["text"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), labelcolor=t["text"],
              edgecolor=t["border"], fontsize=7, framealpha=0)
    return ax


def value_3d(ax, predictions: pd.DataFrame, palette=None):
    """Cost x ownership x predicted points. ax MUST be projection='3d'."""
    t = _p(palette)
    df = predictions[predictions["status_ok"]].dropna(
        subset=["now_cost_m", "pred_points_adj", "selected_by_percent"]
    ).copy()
    if df.empty:
        return _empty(ax, "No data", t)
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    colors = df["position"].map(POSITION_COLORS).fillna(t["text_dim"])
    ax.scatter(df["now_cost_m"], df["selected_by_percent"], df["pred_points_adj"],
               c=colors, s=18, alpha=0.7, depthshade=True)
    ax.set_xlabel("Cost (GBP m)")
    ax.set_ylabel("Owned %")
    ax.set_zlabel("Predicted pts")
    ax.set_title("Cost x Ownership x Predicted Points")
    style_axes(ax, t, three_d=True)
    return ax


def fixture_heatmap(ax, fixture_df: pd.DataFrame, palette=None, transparent=False):
    """fixture_df: rows=team_name, cols=GW+1..GW+N, values=FDR 1-5."""
    t = _p(palette)
    if fixture_df is None or fixture_df.empty:
        return _empty(ax, "No fixture data", t)
    data = fixture_df.values.astype(float)
    cmap = LinearSegmentedColormap.from_list("fdr", [t["accent2"], "#facc15", t["accent"]])
    im = ax.imshow(data, cmap=cmap, vmin=1, vmax=5, aspect="auto")
    ax.set_yticks(range(len(fixture_df.index)))
    ax.set_yticklabels(fixture_df.index, fontsize=7)
    ax.set_xticks(range(len(fixture_df.columns)))
    ax.set_xticklabels(fixture_df.columns, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                        color="#ffffff", fontsize=7)
    ax.set_title("Fixture difficulty (greener = easier)")
    style_axes(ax, t, transparent=transparent)
    ax.grid(False)
    return im


# ------------------------------------------------------------- transfers ---

def squad_value_bars(ax, squad: pd.DataFrame, palette=None, transparent=False):
    t = _p(palette)
    if squad is None or squad.empty or "buy_price_m" not in squad:
        return _empty(ax, "No squad value data", t)
    df = squad.sort_values("profit_m")
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["buy_price_m"], width, label="Bought", color=t["text_dim"])
    ax.bar(x + width / 2, df["now_cost_m"], width, label="Now", color=t["accent2"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["web_name"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("GBP m")
    ax.set_title("Squad value: bought vs current price")
    ax.legend(labelcolor=t["text"], edgecolor=t["border"], fontsize=8, framealpha=0)
    style_axes(ax, t, transparent=transparent)
    return ax


def transfer_gains(ax, transfers: pd.DataFrame, palette=None, transparent=False):
    t = _p(palette)
    if transfers is None or transfers.empty:
        return _empty(ax, "No transfer suggestions", t)
    df = transfers.drop_duplicates("in").nlargest(10, "pred_gain").sort_values("pred_gain")
    labels = [f"{r['out']} -> {r['in']}" for _, r in df.iterrows()]
    ax.barh(labels, df["pred_gain"], color=t["accent2"])
    ax.set_xlabel("Predicted point gain")
    ax.set_title("Best transfer upgrades available")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


def price_momentum_scatter(ax, predictions: pd.DataFrame, palette=None, transparent=False):
    t = _p(palette)
    if predictions is None or predictions.empty:
        return _empty(ax, "No predictions", t)
    df = predictions.dropna(subset=["price_signal", "selected_by_percent"]).copy()
    if df.empty:
        return _empty(ax, "No price signal data", t)
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    colors = np.where(df["price_signal"] >= 3, t["accent2"],
                      np.where(df["price_signal"] <= -3, t["accent"], t["text_dim"]))
    ax.scatter(df["selected_by_percent"], df["price_signal"], s=16, c=colors, alpha=0.7)
    risers = df[df["price_signal"] >= 3].nsmallest(8, "selected_by_percent")
    for _, row in risers.iterrows():
        ax.annotate(row["web_name"], (row["selected_by_percent"], row["price_signal"]),
                    color=t["text"], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color=t["border"], linewidth=0.8)
    ax.set_xlabel("Owned %")
    ax.set_ylabel("Price momentum signal")
    ax.set_title("Price momentum vs ownership")
    style_axes(ax, t, transparent=transparent)
    return ax


# ------------------------------------------------------------ mini league ---

def league_standings_bar(ax, standings: pd.DataFrame, my_entry_name: str,
                         palette=None, transparent=False):
    t = _p(palette)
    if standings is None or standings.empty:
        return _empty(ax, "No standings", t)
    df = standings.sort_values("total")
    colors = [t["accent2"] if n == my_entry_name else t["series_1"] for n in df["entry_name"]]
    ax.barh(df["entry_name"], df["total"], color=colors)
    ax.set_xlabel("Total points")
    ax.set_title("League standings")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


def ownership_chart(ax, ownership: pd.DataFrame, palette=None, transparent=False):
    t = _p(palette)
    if ownership is None or ownership.empty:
        return _empty(ax, "No ownership data", t)
    col = "owned_by_n" if "owned_by_n" in ownership.columns else "owned_pct"
    df = ownership.head(12).sort_values(col)
    ax.barh(df["web_name"], df[col], color=SERIES[0])
    ax.set_xlabel("Managers owning" if col == "owned_by_n" else "Owned %")
    ax.set_title("Most-owned players in your league")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


def captaincy_bar(ax, captains: pd.DataFrame, palette=None, transparent=False):
    """Sorted horizontal bar, deliberately NOT a pie: a 6-colour pie cycle gave
    the 7th captain the same hue as the 1st, and slice angles are hard to
    compare. Bars with raw counts are exact. Tail folds into 'Other' rather
    than recycling hues."""
    t = _p(palette)
    if captains is None or captains.empty:
        return _empty(ax, "No captain data", t)
    col = "captained_by_n" if "captained_by_n" in captains.columns else captains.columns[-1]
    df = captains.sort_values(col, ascending=False)

    top = df.head(8)
    tail_n = int(df[col][8:].sum()) if len(df) > 8 else 0
    names = list(top["web_name"])
    counts = [int(v) for v in top[col]]
    if tail_n:
        names.append(f"Other ({len(df) - 8})")
        counts.append(tail_n)

    y = list(range(len(names)))[::-1]
    colors = [t["series_1"]] * len(names)
    if tail_n:
        colors[-1] = t["series_mute"]
    ax.barh(y, counts, color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Managers captaining")
    ax.set_title("Captaincy choices this GW")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    for yi, c in zip(y, counts):
        ax.text(c, yi, f" {c}", va="center", ha="left", color=t["text_dim"], fontsize=8.5)
    ax.margins(x=0.12)
    return ax


def rank_progression(ax, progression: pd.DataFrame, my_entry_name: str,
                     palette=None, transparent=False):
    """progression: index=gameweek, one column per manager, values=cumulative pts."""
    t = _p(palette)
    if progression is None or progression.empty:
        return _empty(ax, "No history yet", t)
    for col in progression.columns:
        if col == my_entry_name:
            continue
        ax.plot(progression.index, progression[col], color=t["border"],
                linewidth=1, alpha=0.55)
    if my_entry_name in progression.columns:
        ax.plot(progression.index, progression[my_entry_name], color=t["accent2"],
                linewidth=2.5, label=my_entry_name, marker="o", markersize=3)
        ax.legend(labelcolor=t["text"], edgecolor=t["border"], fontsize=8, framealpha=0)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative total points")
    ax.set_title("League points progression")
    style_axes(ax, t, transparent=transparent)
    return ax


def league_form_bar(ax, form: pd.DataFrame, last_n: int, palette=None, transparent=False):
    t = _p(palette)
    if form is None or form.empty:
        return _empty(ax, "No form data", t)
    df = form.sort_values("recent_points")
    ax.barh(df["entry_name"], df["recent_points"], color=SERIES[2])
    ax.set_xlabel(f"Points, last {last_n} GWs")
    ax.set_title(f"League form (last {last_n} gameweeks)")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


def bench_points_lost_bar(ax, per_manager_stats: pd.DataFrame, palette=None, transparent=False):
    """Total points left on the bench this season, per manager -- the
    league_projection.banter_stats() output existed before the Nothing
    redesign but had no chart (or any UI at all) wired to it. Red = worse
    than the league median (more points wasted), green = better."""
    t = _p(palette)
    if per_manager_stats is None or per_manager_stats.empty:
        return _empty(ax, "No bench data", t)
    df = per_manager_stats.sort_values("total_bench_points_lost")
    med = df["total_bench_points_lost"].median()
    colors = [t["bad"] if v > med else t["good"] for v in df["total_bench_points_lost"]]
    ax.barh(df["entry_name"], df["total_bench_points_lost"], color=colors)
    ax.axvline(med, color=t["text_dim"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Total points left on the bench")
    ax.set_title("Bench points lost this season")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


def squad_value_growth_bar(ax, per_manager_stats: pd.DataFrame, palette=None, transparent=False):
    """Squad value growth since GW1 -- who's actually been good at price
    rises vs who's sat still. Same banter_stats() source as the bench chart."""
    t = _p(palette)
    if per_manager_stats is None or per_manager_stats.empty:
        return _empty(ax, "No value data", t)
    df = per_manager_stats.sort_values("squad_value_growth")
    colors = [t["good"] if v >= 0 else t["bad"] for v in df["squad_value_growth"]]
    ax.barh(df["entry_name"], df["squad_value_growth"], color=colors)
    ax.axvline(0, color=t["text_dim"], linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Squad value growth since GW1 (GBPm)")
    ax.set_title("Who's actually grown their squad value")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax


# ------------------------------------------------------------------ model ---

def backtest_mae(ax, backtest_df: pd.DataFrame, palette=None, transparent=False):
    """Model advantage over naive, per gameweek.

    Two near-parallel lines that never cross hide the thing the reader wants,
    which is the GAP. Plot (naive - model) as columns off a zero rule, and
    break at fold boundaries so folds read as separate experiments rather than
    one continuous series.
    """
    t = _p(palette)
    if backtest_df is None or backtest_df.empty:
        return _empty(ax, "No backtest results", t)
    df = backtest_df.copy()
    if "mae_naive" not in df or "mae_model" not in df:
        return _empty(ax, "backtest columns missing", t)

    df["gain"] = df["mae_naive"] - df["mae_model"]
    colors = [t["good"] if g > 0 else t["bad"] for g in df["gain"]]
    x = range(len(df))
    ax.bar(x, df["gain"], color=colors, width=0.72)
    ax.axhline(0, color=t["text_dim"], linewidth=1)

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(g) for g in df["GW"]], fontsize=7.5)
    ax.set_xlabel("Gameweek predicted")
    ax.set_ylabel("MAE improvement vs naive")

    wins = int((df["gain"] > 0).sum())
    ax.set_title(f"Model beats naive in {wins}/{len(df)} gameweeks "
                 f"(mean {df['gain'].mean():.3f} MAE)")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="y", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="x", visible=False)

    if "fold_trained_before_gw" in df:
        folds = df["fold_trained_before_gw"].to_numpy()
        for i in range(1, len(folds)):
            if folds[i] != folds[i - 1]:
                ax.axvline(i - 0.5, color=t["border"], linewidth=1)
    return ax


def feature_importance_bar(ax, importance: pd.Series, palette=None, transparent=False):
    t = _p(palette)
    if importance is None or importance.empty:
        return _empty(ax, "No model loaded", t)
    df = importance.sort_values()
    ax.barh(df.index, df.values, color=SERIES[0])
    ax.set_xlabel("Gain")
    ax.set_title("Feature importance")
    style_axes(ax, t, transparent=transparent)
    ax.grid(axis="x", color=t["border"], linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)
    return ax
