"""FPL scoring rules (2025-26) and the composition step of the component model.

The component model predicts each underlying event rate for a player-fixture
(expected goals, assists, clean-sheet probability, goals-conceded rate, saves,
bonus, defensive-contribution probability, minutes bucket) and this module
turns those into an expected-points number by applying the actual FPL scoring
matrix -- rather than regressing points directly.

Why this and not a single regressor:
  - The 2025-26 "defensive contribution" points (DEF: 10+ CBIT -> +2;
    MID/FWD: 12+ CBIT+recoveries -> +2) are a rule change with ~0 rows of
    training history. A direct points regressor cannot learn them; here they
    enter as an explicit +2 * P(threshold) term.
  - Double gameweeks: a direct regressor trained on one-fixture rows has no
    way to express "this player has two fixtures". Here we compose per fixture
    and sum.
  - Interpretability: every predicted point traces back to a component.

All rates below are conditioned on the player STARTING (>=60 min). The
minutes model supplies P(start) / P(cameo) / P(unused) and this module blends:
a cameo is credited a fraction of a starter's attacking rate (subs do score,
but less), controlled by the SUB_* constants.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --- FPL scoring matrix -------------------------------------------------------
GOAL_POINTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
DC_POINTS = 2                      # defensive-contribution bonus
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}   # GKP not eligible
SAVES_PER_POINT = 3
GC_PER_DEDUCTION = 2              # -1 per 2 conceded, GKP/DEF only
YELLOW_POINTS = -1
RED_POINTS = -3

# --- cameo (1-59 min) credit relative to a start ----------------------------
# a sub gets less pitch time and usually comes on with the game state settled;
# empirically sub goal/assist involvement runs ~35-45% of a starter's rate.
SUB_GOAL_SCALE = 0.40
SUB_ASSIST_SCALE = 0.40
SUB_BONUS_SCALE = 0.25
# a cameo cannot realistically hit a defensive-contribution threshold or a
# saves/clean-sheet milestone, so those are credited only on a true start.

APPEARANCE_CAMEO = 1
APPEARANCE_START = 2


def expected_floor_div(mu, divisor: int, kmax: int = 20) -> np.ndarray:
    """E[ floor(X / divisor) ] for X ~ Poisson(mu), evaluated by summation.

    Used for the two FPL rules that pay per fixed block of events:
      - goals-conceded: -1 for every ``GC_PER_DEDUCTION`` conceded
      - saves: +1 for every ``SAVES_PER_POINT`` saves
    A closed form exists but the truncated sum is exact to float precision for
    the mu range that occurs here (< 6) and is trivially vectorised.
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-9, None)
    ks = np.arange(0, kmax + 1)
    # log pmf for numerical stability, shape (kmax+1, *mu.shape)
    log_pmf = (ks[:, None] * np.log(mu)[None, :] - mu[None, :]
               - np.array([math.lgamma(k + 1) for k in ks])[:, None])
    pmf = np.exp(log_pmf)
    weights = (ks // divisor).astype(float)[:, None]
    return (pmf * weights).sum(axis=0)


def compose_expected_points(df: pd.DataFrame, *, if_starts: bool = False) -> pd.Series:
    """Compose per-fixture expected points from component predictions.

    Required columns in ``df``:
      position            GKP/DEF/MID/FWD
      p_cameo, p_start    P(1-59 min), P(>=60 min)   (ignored if if_starts)
      lam_goals_start     E[goals | start]           (Poisson mean)
      lam_assists_start   E[assists | start]
      exp_bonus_start     E[bonus points | start]
      p_cs_team           P(team keeps a clean sheet)
      mu_gc_team          E[team goals conceded]
      lam_saves_start     E[saves | start]           (GKP; 0/NaN otherwise)
      p_dc               P(defensive-contribution threshold | start)
      yc_rate, rc_rate   rolling per-appearance card rates

    ``if_starts=True`` forces P(start)=1, P(cameo)=0: the "if he plays 90"
    number the UI shows next to the risk-discounted expectation.
    """
    pos = df["position"].astype(str).values
    n = len(df)

    if if_starts:
        p_start = np.ones(n)
        p_cameo = np.zeros(n)
    else:
        p_start = pd.to_numeric(df["p_start"], errors="coerce").fillna(0.0).values
        p_cameo = pd.to_numeric(df["p_cameo"], errors="coerce").fillna(0.0).values

    def col(name, default=0.0):
        if name not in df.columns:
            return np.full(n, default, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(default).values

    lam_goals = col("lam_goals_start")
    lam_assists = col("lam_assists_start")
    exp_bonus = col("exp_bonus_start")
    p_cs_team = np.clip(col("p_cs_team"), 0.0, 1.0)
    mu_gc_team = np.clip(col("mu_gc_team"), 0.0, None)
    lam_saves = col("lam_saves_start")
    p_dc = np.clip(col("p_dc"), 0.0, 1.0)
    yc_rate = col("yc_rate")
    rc_rate = col("rc_rate")

    goal_val = np.array([GOAL_POINTS.get(p, 5) for p in pos], dtype=float)
    cs_val = np.array([CLEAN_SHEET_POINTS.get(p, 0) for p in pos], dtype=float)
    is_gkp_def = np.isin(pos, ["GKP", "DEF"])
    is_gkp = pos == "GKP"
    dc_eligible = np.isin(pos, ["DEF", "MID", "FWD"])

    # effective playing weights
    played = p_start + p_cameo
    goal_weight = p_start + SUB_GOAL_SCALE * p_cameo
    assist_weight = p_start + SUB_ASSIST_SCALE * p_cameo
    bonus_weight = p_start + SUB_BONUS_SCALE * p_cameo

    pts = np.zeros(n)
    # appearance
    pts += APPEARANCE_CAMEO * p_cameo + APPEARANCE_START * p_start
    # attacking returns
    pts += goal_val * lam_goals * goal_weight
    pts += ASSIST_POINTS * lam_assists * assist_weight
    # clean sheet (needs a full start)
    pts += cs_val * p_cs_team * p_start
    # goals-conceded deduction, GKP/DEF only, needs a full start
    gc_deductions = expected_floor_div(mu_gc_team, GC_PER_DEDUCTION)
    pts += np.where(is_gkp_def, -gc_deductions * p_start, 0.0)
    # saves, GKP only
    save_points = expected_floor_div(lam_saves, SAVES_PER_POINT)
    pts += np.where(is_gkp, save_points * p_start, 0.0)
    # bonus
    pts += exp_bonus * bonus_weight
    # defensive contribution (2025-26)
    pts += np.where(dc_eligible, DC_POINTS * p_dc * p_start, 0.0)
    # cards
    pts += YELLOW_POINTS * yc_rate * played + RED_POINTS * rc_rate * played

    return pd.Series(pts, index=df.index, name="pred_points_if_starts" if if_starts else "pred_points")


def actual_points_check(row: pd.Series) -> int:
    """Recompute a historical row's FPL points from its raw stat line, as a
    unit-test oracle for the scoring constants above. Not used in production;
    handy in a notebook to confirm GOAL_POINTS/CLEAN_SHEET_POINTS etc. match
    the season being looked at (the DC rule only exists from 2025-26)."""
    pos = row["position"]
    pts = 0
    mins = row.get("minutes", 0) or 0
    if mins >= 60:
        pts += 2
    elif mins > 0:
        pts += 1
    pts += GOAL_POINTS.get(pos, 5) * int(row.get("goals_scored", 0) or 0)
    pts += ASSIST_POINTS * int(row.get("assists", 0) or 0)
    if mins >= 60 and int(row.get("clean_sheets", 0) or 0):
        pts += CLEAN_SHEET_POINTS.get(pos, 0)
    if pos in ("GKP", "DEF") and mins >= 60:
        pts -= int((row.get("goals_conceded", 0) or 0)) // GC_PER_DEDUCTION
    if pos == "GKP":
        pts += int((row.get("saves", 0) or 0)) // SAVES_PER_POINT
        pts += 5 * int(row.get("penalties_saved", 0) or 0)
    pts += int(row.get("bonus", 0) or 0)
    pts -= 2 * int(row.get("penalties_missed", 0) or 0)
    pts -= 2 * int(row.get("own_goals", 0) or 0)
    pts += YELLOW_POINTS * int(row.get("yellow_cards", 0) or 0)
    pts += RED_POINTS * int(row.get("red_cards", 0) or 0)
    dc = row.get("defensive_contribution", 0) or 0
    if pos in DC_THRESHOLD and dc >= DC_THRESHOLD[pos]:
        pts += DC_POINTS
    return pts
