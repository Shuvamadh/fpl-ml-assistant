"""Qt bindings for the shared chart core.

The drawing logic lives in src/charts_core.py, which imports no Qt and no
Streamlit. This file does two things and nothing else: build/clear a Qt canvas,
and hand its Axes plus the live theme palette to the core.

Previously every chart was implemented here against a ChartCanvas, which made
them unimportable from the Streamlit app -- so that app hand-rewrote two of
them and simply went without the other eleven. Anything added to charts_core
now appears in both frontends automatically.

Every draw_* name below is preserved so existing call sites in main_window.py
and chart_carousel.py keep working unchanged.
"""
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import charts_core
import mpl_style
import theme


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, width=6, height=4.5):
        fig = Figure(figsize=(width, height), dpi=110)
        super().__init__(fig)
        self.setMinimumHeight(380)


def _prep(canvas, projection=None):
    """Clear the canvas and return (theme palette, fresh Axes).

    Palette is read at call time, not import time, so redrawing after a theme
    switch picks up the new colours -- same contract as before.
    """
    fig = canvas.figure
    fig.clear()
    t = theme.current()
    mpl_style.apply(t)
    fig.set_facecolor(t["panel"])
    ax = fig.add_subplot(111, projection=projection) if projection else fig.add_subplot(111)
    return t, ax


# ---------------------------------------------------------------- players ---

def draw_xi_points_bar(canvas, xi):
    t, ax = _prep(canvas)
    charts_core.xi_points_bar(ax, xi, palette=t)
    canvas.draw()


def draw_captain_uncertainty(canvas, candidates, mae):
    t, ax = _prep(canvas)
    charts_core.captain_uncertainty(ax, candidates, mae, palette=t)
    canvas.draw()


def draw_value_scatter(canvas, predictions):
    t, ax = _prep(canvas)
    charts_core.value_scatter(ax, predictions, palette=t)
    canvas.draw()


def draw_differential_scatter(canvas, predictions, own_thresh=5.0):
    t, ax = _prep(canvas)
    charts_core.differential_scatter(ax, predictions, own_thresh, palette=t)
    canvas.draw()


def draw_radar_comparison(canvas, predictions, players=None):
    t, ax = _prep(canvas, projection="polar")
    charts_core.radar_comparison(ax, predictions, players, palette=t)
    canvas.draw()


def draw_3d_value(canvas, predictions):
    t, ax = _prep(canvas, projection="3d")
    charts_core.value_3d(ax, predictions, palette=t)
    canvas.draw()


def draw_fixture_heatmap(canvas, fixture_df):
    t, ax = _prep(canvas)
    im = charts_core.fixture_heatmap(ax, fixture_df, palette=t)
    if im is not None and hasattr(im, "get_cmap"):
        canvas.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    canvas.draw()


# ------------------------------------------------------------- transfers ---

def draw_squad_value_bars(canvas, squad):
    t, ax = _prep(canvas)
    charts_core.squad_value_bars(ax, squad, palette=t)
    canvas.draw()


def draw_transfer_gains(canvas, transfers):
    t, ax = _prep(canvas)
    charts_core.transfer_gains(ax, transfers, palette=t)
    canvas.draw()


def draw_price_momentum_scatter(canvas, predictions):
    t, ax = _prep(canvas)
    charts_core.price_momentum_scatter(ax, predictions, palette=t)
    canvas.draw()


# ------------------------------------------------------------ mini league ---

def draw_league_standings_bar(canvas, standings, my_entry_name):
    t, ax = _prep(canvas)
    charts_core.league_standings_bar(ax, standings, my_entry_name, palette=t)
    canvas.draw()


def draw_ownership_chart(canvas, ownership):
    t, ax = _prep(canvas)
    charts_core.ownership_chart(ax, ownership, palette=t)
    canvas.draw()


def draw_captaincy_pie(canvas, captains):
    """Name kept for existing call sites; renders sorted bars, not a pie.
    See charts_core.captaincy_bar for why."""
    t, ax = _prep(canvas)
    charts_core.captaincy_bar(ax, captains, palette=t)
    canvas.draw()


def draw_league_form_bar(canvas, form, last_n=4):
    t, ax = _prep(canvas)
    charts_core.league_form_bar(ax, form, last_n, palette=t)
    canvas.draw()


def draw_rank_progression(canvas, progression, my_entry_name):
    t, ax = _prep(canvas)
    charts_core.rank_progression(ax, progression, my_entry_name, palette=t)
    canvas.draw()


# ------------------------------------------------------------------ model ---

def draw_backtest_mae(canvas, backtest_df):
    t, ax = _prep(canvas)
    charts_core.backtest_mae(ax, backtest_df, palette=t)
    canvas.draw()


def draw_feature_importance(canvas, importance):
    t, ax = _prep(canvas)
    charts_core.feature_importance_bar(ax, importance, palette=t)
    canvas.draw()
