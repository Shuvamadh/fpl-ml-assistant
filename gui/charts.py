"""Matplotlib figures embedded in the Qt UI. Every draw_* function reads
colors from theme.current() at call time, so re-drawing after a theme switch
picks up the new palette."""
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

import mpl_style
import theme

def _accents(n: int = 4):
    """Data colors. NOT the chrome accents -- GUI_PLAN 6.1 is explicit that
    `accent`/`accent2` are chrome only, and the old cycling list is what made
    the 7th slice the same hue as the 1st."""
    return mpl_style.series(theme.current(), n)


def _style_axes(ax, three_d=False):
    t = theme.current()
    ax.set_facecolor(t["panel"])
    ax.figure.set_facecolor(t["panel"])
    ax.tick_params(colors=t["text_dim"], labelsize=8)
    if not three_d:
        for spine in ax.spines.values():
            spine.set_color(t["border"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=t["border"], linewidth=0.5, alpha=0.5)
    else:
        ax.xaxis.pane.set_facecolor(t["panel"])
        ax.yaxis.pane.set_facecolor(t["panel"])
        ax.zaxis.pane.set_facecolor(t["panel"])
        ax.grid(color=t["border"], linewidth=0.3, alpha=0.4)
    ax.xaxis.label.set_color(t["text_dim"])
    ax.yaxis.label.set_color(t["text_dim"])
    ax.title.set_color(t["text"])


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, width=6, height=4.5):
        fig = Figure(figsize=(width, height), dpi=110)
        super().__init__(fig)
        self.setMinimumHeight(380)


def _prep(canvas: ChartCanvas, three_d=False):
    fig = canvas.figure
    fig.clear()
    t = theme.current()
    mpl_style.apply(t)
    fig.set_facecolor(t["panel"])
    ax = fig.add_subplot(111, projection="3d") if three_d else fig.add_subplot(111)
    return fig, ax


# ---------------------------------------------------------------- players ---

def draw_xi_points_bar(canvas: ChartCanvas, xi: pd.DataFrame):
    fig, ax = _prep(canvas)
    t = theme.current()
    df = xi.sort_values("pred_points_adj")
    pos_colors = {"GKP": t["accent2"], "DEF": t["accent"], "MID": "#38bdf8", "FWD": "#facc15"}
    colors = [pos_colors.get(p, t["accent2"]) for p in df["position"]]
    ax.barh(df["web_name"], df["pred_points_adj"], color=colors)
    ax.set_xlabel("Predicted points (fixture-adjusted)")
    ax.set_title("Starting XI — predicted points")
    _style_axes(ax)
    canvas.draw()


def draw_value_scatter(canvas: ChartCanvas, predictions: pd.DataFrame):
    fig, ax = _prep(canvas)
    t = theme.current()
    df = predictions[predictions["status_ok"]].dropna(subset=["now_cost_m", "pred_points_adj"])
    ax.scatter(df["now_cost_m"], df["pred_points_adj"], s=14, alpha=0.5, color=t["text_dim"])
    top_value = df.sort_values("value_ratio", ascending=False).head(12)
    ax.scatter(top_value["now_cost_m"], top_value["pred_points_adj"], s=40, color=t["accent2"], zorder=3)
    for _, row in top_value.iterrows():
        ax.annotate(row["web_name"], (row["now_cost_m"], row["pred_points_adj"]),
                    color=t["text"], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cost (£m)")
    ax.set_ylabel("Predicted points")
    ax.set_title("Value hunting: cost vs predicted points")
    _style_axes(ax)
    canvas.draw()


def draw_radar_comparison(canvas: ChartCanvas, predictions: pd.DataFrame, players: list[str] | None = None):
    """Spider chart comparing top players on normalised attributes."""
    fig = canvas.figure
    fig.clear()
    t = theme.current()
    fig.set_facecolor(t["panel"])

    df = predictions[predictions["status_ok"]].copy()
    if players:
        df = df[df["web_name"].isin(players)]
    else:
        df = df.sort_values("pred_points_adj", ascending=False).head(5)

    attrs = ["pred_points_adj", "expected_goal_involvements", "ict_index" if "ict_index" in df else "form",
             "value_ratio", "form"]
    attrs = [a for a in dict.fromkeys(attrs) if a in df.columns]
    labels = {"pred_points_adj": "Pred Pts", "expected_goal_involvements": "xGI",
              "ict_index": "ICT", "value_ratio": "Value", "form": "Form"}

    norm = df[attrs].apply(pd.to_numeric, errors="coerce").fillna(0)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(attrs), endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(111, projection="polar")
    colors = _accents()
    for i, (_, row) in enumerate(df.iterrows()):
        values = norm.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=row["web_name"])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([labels.get(a, a) for a in attrs], color=t["text"], fontsize=8)
    ax.set_yticklabels([])
    ax.set_facecolor(t["panel"])
    ax.spines["polar"].set_color(t["border"])
    ax.grid(color=t["border"], alpha=0.5)
    ax.set_title("Player comparison (normalised)", color=t["text"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), facecolor=t["panel"],
              labelcolor=t["text"], edgecolor=t["border"], fontsize=7)
    canvas.draw()


def draw_3d_value(canvas: ChartCanvas, predictions: pd.DataFrame):
    """3D: cost x predicted points x ownership -- spot undervalued, low-owned gems."""
    fig, ax = _prep(canvas, three_d=True)
    t = theme.current()
    df = predictions[predictions["status_ok"]].dropna(
        subset=["now_cost_m", "pred_points_adj", "selected_by_percent"]
    ).copy()
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)

    pos_colors = {"GKP": t["accent2"], "DEF": t["accent"], "MID": "#38bdf8", "FWD": "#facc15"}
    colors = df["position"].map(pos_colors).fillna(t["text_dim"])

    ax.scatter(df["now_cost_m"], df["selected_by_percent"], df["pred_points_adj"],
               c=colors, s=18, alpha=0.7, depthshade=True)
    ax.set_xlabel("Cost (£m)")
    ax.set_ylabel("Owned %")
    ax.set_zlabel("Predicted pts")
    ax.set_title("Cost x Ownership x Predicted Points")
    _style_axes(ax, three_d=True)
    canvas.draw()


def draw_fixture_heatmap(canvas: ChartCanvas, fixture_df: pd.DataFrame):
    """fixture_df: rows=team_name, cols=next fixture 1..N, values=FDR (1-5)."""
    fig, ax = _prep(canvas)
    t = theme.current()
    data = fixture_df.values.astype(float)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("fdr", [t["accent2"], "#facc15", t["accent"]])
    im = ax.imshow(data, cmap=cmap, vmin=1, vmax=5, aspect="auto")
    ax.set_yticks(range(len(fixture_df.index)))
    ax.set_yticklabels(fixture_df.index, fontsize=7)
    ax.set_xticks(range(len(fixture_df.columns)))
    ax.set_xticklabels(fixture_df.columns, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", color=t["bg"], fontsize=7)
    ax.set_title("Fixture difficulty (next 5 GWs, greener = easier)")
    _style_axes(ax)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    canvas.draw()


# ------------------------------------------------------------- transfers ---

def draw_squad_value_bars(canvas: ChartCanvas, squad: pd.DataFrame):
    fig, ax = _prep(canvas)
    t = theme.current()
    df = squad.sort_values("profit_m")
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["buy_price_m"], width, label="Bought", color=t["text_dim"])
    ax.bar(x + width / 2, df["now_cost_m"], width, label="Now", color=t["accent2"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["web_name"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("£m")
    ax.set_title("Squad value: bought vs current price")
    ax.legend(facecolor=t["panel"], labelcolor=t["text"], edgecolor=t["border"], fontsize=8)
    _style_axes(ax)
    canvas.draw()


def draw_transfer_gains(canvas: ChartCanvas, transfers: pd.DataFrame):
    fig, ax = _prep(canvas)
    t = theme.current()
    if transfers is None or transfers.empty:
        ax.text(0.5, 0.5, "No transfer suggestions", ha="center", va="center", color=t["text_dim"])
        _style_axes(ax)
        canvas.draw()
        return
    df = transfers.drop_duplicates("in").nlargest(10, "pred_gain").sort_values("pred_gain")
    labels = [f"{r['out']} -> {r['in']}" for _, r in df.iterrows()]
    ax.barh(labels, df["pred_gain"], color=t["accent2"])
    ax.set_xlabel("Predicted point gain")
    ax.set_title("Best transfer upgrades available")
    _style_axes(ax)
    canvas.draw()


def draw_price_momentum_scatter(canvas: ChartCanvas, predictions: pd.DataFrame):
    fig, ax = _prep(canvas)
    t = theme.current()
    df = predictions.dropna(subset=["price_signal", "selected_by_percent"]).copy()
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
    ax.set_title("Price momentum vs ownership (green=rising, pink=falling)")
    _style_axes(ax)
    canvas.draw()


# ------------------------------------------------------------ mini league ---

def draw_league_standings_bar(canvas: ChartCanvas, standings: pd.DataFrame, my_entry_name: str):
    fig, ax = _prep(canvas)
    t = theme.current()
    df = standings.sort_values("total")
    colors = [t["accent2"] if n == my_entry_name else t["accent"] for n in df["entry_name"]]
    ax.barh(df["entry_name"], df["total"], color=colors)
    ax.set_xlabel("Total points")
    ax.set_title("League standings")
    _style_axes(ax)
    canvas.draw()


def draw_ownership_chart(canvas: ChartCanvas, ownership: pd.DataFrame):
    fig, ax = _prep(canvas)
    t = theme.current()
    df = ownership.head(12).sort_values("owned_by_n")
    ax.barh(df["web_name"], df["owned_by_n"], color=_accents()[2])
    ax.set_xlabel("Managers owning")
    ax.set_title("Most-owned players in your league")
    _style_axes(ax)
    canvas.draw()


def draw_captaincy_pie(canvas: ChartCanvas, captains: pd.DataFrame):
    """Sorted horizontal bar, not a pie (GUI_PLAN 3).

    The old pie cycled a 6-color list, so a 7th captain got the same hue as the
    1st, and slice angles are hard to compare. Bars with raw counts are exact.
    Kept under the original name so existing call sites keep working.
    """
    fig, ax = _prep(canvas)
    t = theme.current()
    df = captains.sort_values("captained_by_n", ascending=False)

    # fold the tail into "Other" instead of cycling hues
    top = df.head(8)
    tail_n = int(df["captained_by_n"][8:].sum()) if len(df) > 8 else 0
    names = list(top["web_name"])
    counts = [int(v) for v in top["captained_by_n"]]
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
    # grid parallel to the bars is noise; only the value axis carries it
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for yi, c in zip(y, counts):
        ax.text(c, yi, f" {c}", va="center", ha="left",
                color=t["text_dim"], fontsize=8.5)
    ax.margins(x=0.12)
    canvas.draw()


def draw_rank_progression(canvas: ChartCanvas, progression: pd.DataFrame, my_entry_name: str):
    """progression: columns = event (GW), rows = one line per manager, values = cumulative total pts."""
    fig, ax = _prep(canvas)
    t = theme.current()
    colors = _accents()
    for i, col in enumerate(progression.columns):
        if col == my_entry_name:
            continue
        ax.plot(progression.index, progression[col], color=t["border"], linewidth=1, alpha=0.6)
    if my_entry_name in progression.columns:
        ax.plot(progression.index, progression[my_entry_name], color=t["accent2"], linewidth=2.5,
                 label=my_entry_name, marker="o", markersize=3)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative total points")
    ax.set_title("League points progression")
    ax.legend(facecolor=t["panel"], labelcolor=t["text"], edgecolor=t["border"], fontsize=8)
    _style_axes(ax)
    canvas.draw()


# ------------------------------------------------------------------ model ---

def draw_backtest_mae(canvas: ChartCanvas, backtest_df: pd.DataFrame):
    """Model advantage over the naive baseline, per gameweek.

    GUI_PLAN: the old version drew two near-parallel lines that never cross and
    ran an unbroken line straight through fold boundaries, implying continuity
    where a different model actually takes over. What the reader wants is the
    GAP, so plot (naive - model) as columns off a zero rule, and break the
    series at each fold so the folds read as separate experiments.
    """
    fig, ax = _prep(canvas)
    t = theme.current()
    df = backtest_df.copy()
    if "mae_naive" not in df or "mae_model" not in df:
        ax.text(0.5, 0.5, "backtest columns missing", ha="center", va="center",
                color=t["text_dim"], transform=ax.transAxes)
        canvas.draw()
        return

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
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    # fold boundaries: the folds are separate models, so mark where they change
    if "fold_trained_before_gw" in df:
        folds = df["fold_trained_before_gw"].to_numpy()
        for i in range(1, len(folds)):
            if folds[i] != folds[i - 1]:
                ax.axvline(i - 0.5, color=t["border"], linewidth=1, linestyle="-")
    canvas.draw()
