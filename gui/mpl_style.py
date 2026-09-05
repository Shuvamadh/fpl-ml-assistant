"""Matplotlib styling so the charts read as part of the app rather than a
foreign object dropped into it (GUI_PLAN 6.9).

`apply()` must run at the top of every draw_* function, BEFORE the axes are
built, so that switching theme and redrawing picks up the new colors.
"""


def apply(t: dict) -> None:
    """t = theme.current()."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": t["panel"],
        "figure.edgecolor": "none",
        "figure.dpi": 110,
        "savefig.facecolor": t["panel"],
        "axes.facecolor": t["panel"],
        "axes.edgecolor": t["border"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": t["text_dim"],
        "axes.labelsize": 9,
        "axes.labelpad": 8,
        "axes.titlecolor": t["text"],
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t["border"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,
        "grid.linestyle": "-",
        "xtick.color": t["text_dim"],
        "ytick.color": t["text_dim"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "text.color": t["text"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI Variable Text", "Segoe UI", "Inter", "DejaVu Sans"],
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": t["text"],
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "scatter.edgecolors": "none",
        # constrained_layout and tight_layout conflict; every fig.tight_layout()
        # call must be removed wherever this is active.
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
    })


def series(t: dict, n: int = 1) -> list[str]:
    """Categorical color slots in FIXED order.

    Never cycles: with more than 4 categories the tail must fold into "Other"
    or the chart must facet. Cycling is what makes the 7th captain in a pie
    identical in hue to the 1st. Scatter / all-pairs forms cap at 3.
    """
    slots = [t["series_1"], t["series_2"], t["series_3"], t["series_4"]]
    if n > len(slots):
        raise ValueError(
            f"{n} categories: fold the tail into 'Other' or facet -- do not cycle hues"
        )
    return slots[:n]


def emphasis(t: dict, n: int, highlight: int | None = None) -> list[str]:
    """One highlighted mark against a muted field -- the right encoding when a
    chart answers "where am I" rather than "compare these categories"."""
    colors = [t["series_mute"]] * n
    if highlight is not None and 0 <= highlight < n:
        colors[highlight] = t["series_1"]
    return colors
