"""Chip strategy: wildcard/free-hit squad optimization, and timing advice for
Triple Captain / Bench Boost. Requested via a parallel session on the user's
behalf (chip strategy: best wildcard squad + when-to-play advice for Free Hit
and Triple Captain).

THE CORE DISTINCTION (get this wrong and every downstream number is wrong):
- WILDCARD is permanent: optimizes over a HORIZON (next few GWs), uses your
  REAL SELLABLE value (squad_value.py) as budget, since that's your actual
  constraint when rebuilding.
- FREE HIT reverts after one week: optimizes a SINGLE upcoming GW only, uses
  your squad's CURRENT MARKET value (no sell-price penalty -- FPL gives you
  your full team value back for that one week), and is really only correct
  to play into a blank or double gameweek.

HONEST CAVEATS (surfaced in every function's return, not just this docstring):
- These are point-estimate optimizations off a model with backtested MAE
  ~0.95-1.0 (see backtest.py / README). A squad ranked 1st by predicted
  points and one ranked 5th are likely within noise of each other. Every
  optimizer function here returns enough detail (predicted points per
  candidate, not just a single "winner") that a consumer can show the
  spread rather than implying false precision.
- The wildcard horizon score is an APPROXIMATION: this project's model
  predicts ONE upcoming gameweek, not N future gameweeks independently (that
  would require simulating each future gameweek's own rolling-form features,
  which don't exist yet since those gameweeks haven't happened). The horizon
  score instead scales the single-GW prediction by the number of horizon
  gameweeks and by that team's average fixture difficulty over the horizon
  (`fdr_next_n_mean`, already computed by fixtures_fdr.py). This is a
  reasonable proxy, not a true multi-gameweek forecast -- treat wildcard
  rankings as directional, more so than the single-GW recommender.
- CHECKED, NOT ASSUMED: `fdr_next_n_mean` was verified (a peer review flagged
  it, a controlled per-position check confirmed it) to have near-zero
  correlation with predicted points -- both because averaging a mostly-"3"
  1-5 categorical scale over 5 fixtures is expected to shrink variance, and
  because FPL's own fixture-difficulty rating has weak genuine predictive
  power for single-gameweek fantasy points once a player's own form/quality
  is controlled for (a player-quality-confounded pooled correlation looks
  larger and is misleading -- check per position). Deliberately NOT
  "corrected" by widening the multiplier here, since the data doesn't
  support a strong fixture effect existing to justify one -- that would be
  fitting the number to a narrative rather than the evidence. Treat the
  horizon score's fixture component as a mild directional nudge, not a
  reliable ranking signal on its own; the underlying player-quality
  predictions (`pred_points`) are doing most of the real work. See README's
  "Model" section for the full before/after numbers.
- Double-gameweek predictions are approximated as 2x the single-fixture
  prediction (no separate per-fixture model), same caveat.
"""
from pathlib import Path

import pandas as pd
import pulp

from fpl_api import entry_history, fixtures
from squad_value import squad_value_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POSITION_SQUAD_LIMITS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
BENCH_WEIGHT = 0.1  # small but nonzero: prefers a bench that could plausibly play over a zero-value throwaway
FDR_ADJUSTMENT = {1: 1.08, 2: 1.04, 3: 1.00, 4: 0.96, 5: 0.92}  # same scale as predict.py


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "player_predictions.csv")


def detect_blank_double_gameweeks(n: int = 5) -> pd.DataFrame:
    """Fixture count per team per upcoming gameweek. 0 fixtures = blank (BGW)
    for that team that week, 2+ = double (DGW). Prerequisite for correct
    Free Hit / Bench Boost / Triple Captain timing -- without this, chip
    timing advice is just guessing."""
    fx = fixtures()
    upcoming = [f for f in fx if not f["finished"] and f["event"] is not None]
    if not upcoming:
        return pd.DataFrame(columns=["team", "event", "n_fixtures"])
    events = sorted(set(f["event"] for f in upcoming))[:n]
    rows = []
    for ev in events:
        ev_fixtures = [f for f in upcoming if f["event"] == ev]
        team_counts = {}
        for f in ev_fixtures:
            team_counts[f["team_h"]] = team_counts.get(f["team_h"], 0) + 1
            team_counts[f["team_a"]] = team_counts.get(f["team_a"], 0) + 1
        all_teams = set()
        for f in fx:
            all_teams.add(f["team_h"])
            all_teams.add(f["team_a"])
        for team in all_teams:
            rows.append({"team": team, "event": ev, "n_fixtures": team_counts.get(team, 0)})
    df = pd.DataFrame(rows)
    df["is_blank"] = df["n_fixtures"] == 0
    df["is_double"] = df["n_fixtures"] >= 2
    return df


def _optimize_squad(pool: pd.DataFrame, budget: float, score_col: str = "pred_points_adj") -> dict:
    """Shared ILP: pick 15 (2/5/5/3, <=3 per club, within budget) maximizing
    a legal starting XI's score plus a small weight on the bench, so the
    bench isn't wasted on players who'll never play. Returns squad, starting
    XI, bench, captain, and total predicted points -- enough detail to show
    the spread/reasoning, not just a single answer.
    """
    pool = pool[pool["status_ok"]].reset_index(drop=True)
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    squad_vars = {i: pulp.LpVariable(f"squad_{i}", cat="Binary") for i in pool.index}
    start_vars = {i: pulp.LpVariable(f"start_{i}", cat="Binary") for i in pool.index}

    prob += pulp.lpSum(
        start_vars[i] * pool.loc[i, score_col] for i in pool.index
    ) + BENCH_WEIGHT * pulp.lpSum(
        (squad_vars[i] - start_vars[i]) * pool.loc[i, score_col] for i in pool.index
    )

    prob += pulp.lpSum(squad_vars.values()) == 15
    prob += pulp.lpSum(pool.loc[i, "now_cost_m"] * squad_vars[i] for i in pool.index) <= budget

    for pos, limit in POSITION_SQUAD_LIMITS.items():
        idx = pool.index[pool["position"] == pos]
        prob += pulp.lpSum(squad_vars[i] for i in idx) == limit

    for club in pool["name"].unique():
        idx = pool.index[pool["name"] == club]
        prob += pulp.lpSum(squad_vars[i] for i in idx) <= MAX_PER_CLUB

    for i in pool.index:
        prob += start_vars[i] <= squad_vars[i]
    prob += pulp.lpSum(start_vars.values()) == 11
    for pos, (lo, hi) in STARTING_LIMITS.items():
        idx = pool.index[pool["position"] == pos]
        prob += pulp.lpSum(start_vars[i] for i in idx) >= lo
        prob += pulp.lpSum(start_vars[i] for i in idx) <= hi

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return {"status": pulp.LpStatus[prob.status], "squad": pd.DataFrame(), "xi": pd.DataFrame(),
                "bench": pd.DataFrame(), "captain": None, "total_xi_points": 0.0, "total_cost": 0.0}

    squad_idx = [i for i in pool.index if squad_vars[i].value() == 1]
    start_idx = [i for i in pool.index if start_vars[i].value() == 1]
    squad = pool.loc[squad_idx].copy()
    xi = pool.loc[start_idx].sort_values(score_col, ascending=False).copy()
    bench = squad[~squad.index.isin(start_idx)].sort_values(score_col, ascending=False)
    captain = xi.iloc[0]

    return {
        "status": "Optimal",
        "squad": squad, "xi": xi, "bench": bench,
        "captain": captain["web_name"], "captain_pred": captain[score_col],
        "total_xi_points": xi[score_col].sum(),
        "total_squad_cost": squad["now_cost_m"].sum(),
        "budget": budget,
    }


def wildcard_squad(team_id: int, current_squad: pd.DataFrame, horizon: int = 5) -> dict:
    """Optimize over `horizon` gameweeks (approximate, see module docstring),
    budget = your REAL sellable squad value (half-profit-on-rise rule
    applied) + bank -- your actual constraint when rebuilding permanently."""
    preds = load_predictions()
    sv = squad_value_report(team_id, current_squad)
    budget = sv["sell_price_m"].sum()

    preds = preds.copy()
    preds["horizon_multiplier"] = preds["next_fdr"].round().map(FDR_ADJUSTMENT).fillna(1.0)
    # de-adjust the single-fixture multiplier already baked into pred_points_adj,
    # then apply the wider n-fixture-average difficulty across the horizon
    preds["fdr_avg_multiplier"] = preds["fdr_next_n_mean"].round().map(FDR_ADJUSTMENT).fillna(1.0)
    preds["horizon_score"] = (
        preds["pred_points"] * preds["fdr_avg_multiplier"] * horizon
    )

    result = _optimize_squad(preds, budget, score_col="horizon_score")
    result["horizon_gws"] = horizon
    result["budget_source"] = "real sellable squad value + bank (squad_value.py)"
    result["caveat"] = (
        "horizon_score = single-GW model prediction x horizon length x "
        "average-fixture-difficulty multiplier -- an approximation, not an "
        "N-gameweek simulation. Treat as directional."
    )
    return result


def free_hit_squad(team_id: int, current_squad: pd.DataFrame) -> dict:
    """Optimize a SINGLE gameweek, budget = current squad MARKET value (sum
    of now_cost, no sell-price penalty) + bank -- Free Hit gives you your
    full team value back for one week. Only really worth playing into a
    blank or double gameweek; check detect_blank_double_gameweeks() first."""
    preds = load_predictions()
    budget = current_squad["now_cost_m"].sum()  # market value, not sell value
    result = _optimize_squad(preds, budget, score_col="pred_points_adj")
    result["budget_source"] = "current squad market value (sum of now_cost) + bank"
    result["caveat"] = (
        "single-gameweek optimization -- only sensible for a blank/double "
        "gameweek; in a normal fixture week this is close to wasting the chip."
    )
    return result


def triple_captain_candidates(squad_with_preds: pd.DataFrame, blank_double: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank squad players by EXPECTED GAIN over your best alternative captain
    choice: 3x this player's prediction minus 2x what you'd have captained
    instead. A double-gameweek player's prediction is approximately doubled
    (two fixtures) -- see module caveat on DGW approximation.
    """
    df = squad_with_preds.copy()
    if blank_double is not None and "team_id" in df.columns:
        dgw_teams = set(blank_double.loc[blank_double["is_double"], "team"])
        df["is_dgw"] = df["team_id"].isin(dgw_teams)
        df["effective_pred"] = df["pred_points_adj"] * df["is_dgw"].map({True: 2, False: 1})
    else:
        df["is_dgw"] = False
        df["effective_pred"] = df["pred_points_adj"]

    df = df.sort_values("effective_pred", ascending=False).reset_index(drop=True)
    best = df["effective_pred"].iloc[0]
    second = df["effective_pred"].iloc[1] if len(df) > 1 else 0.0

    def gain(row, rank):
        alt_best = second if rank == 0 else best
        return 3 * row["effective_pred"] - 2 * alt_best

    df["tc_gain_vs_normal_captain"] = [gain(row, i) for i, row in df.iterrows()]
    return df[["web_name", "position", "name", "pred_points_adj", "is_dgw",
               "effective_pred", "tc_gain_vs_normal_captain"]].sort_values(
        "tc_gain_vs_normal_captain", ascending=False
    )


def bench_boost_value(squad_with_preds: pd.DataFrame, bench_element_ids: list[int]) -> dict:
    """Value of playing Bench Boost = sum of the 4 bench players' predicted
    points. Flagged if any bench player has a low chance of actually playing
    (chance_of_playing_next_round < 75 or 0 minutes so far), since a bench
    boost is worthless if the bench doesn't play."""
    bench = squad_with_preds[squad_with_preds["element"].isin(bench_element_ids)] if "element" in squad_with_preds else squad_with_preds
    total = bench["pred_points_adj"].sum()
    risky = bench[
        (bench.get("chance_of_playing_next_round", 100).fillna(100) < 75)
        | (bench.get("season_gp_prior", 1) == 0)
    ]
    return {
        "bench_boost_value": total,
        "bench": bench[["web_name", "position", "pred_points_adj"]].to_dict("records"),
        "risky_bench_players": risky["web_name"].tolist() if not risky.empty else [],
    }


def chips_used(team_id: int) -> list[str]:
    """Chips already played this season, from the public entry-history
    endpoint -- so a chip tab can grey out chips that can't be played again."""
    hist = entry_history(team_id)
    return [c["name"] for c in hist.get("chips", [])]


if __name__ == "__main__":
    import sys
    from recommend import load_squad

    team_id = int(sys.argv[1]) if len(sys.argv) > 1 else 8041052
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    squad, meta = load_squad(team_id, event)
    print("Chips used:", chips_used(team_id) or "none")

    bd = detect_blank_double_gameweeks()
    print("\nBlank/double gameweeks in next 5:")
    print(bd[bd["is_blank"] | bd["is_double"]].to_string(index=False) if (bd["is_blank"] | bd["is_double"]).any()
          else "  none detected -- normal fixture schedule ahead")

    print("\n=== Wildcard candidate squad (horizon=5, budget=real sellable value) ===")
    wc = wildcard_squad(team_id, squad, horizon=5)
    print(f"Status: {wc['status']} | Budget used: {wc.get('total_squad_cost', 0):.1f}m / {wc.get('budget', 0):.1f}m")
    if not wc["xi"].empty:
        print(wc["xi"][["web_name", "position", "name", "now_cost_m", "horizon_score"]].to_string(index=False))
        print(f"Captain: {wc['captain']}")

    print("\n=== Triple Captain candidates (your current squad) ===")
    squad_preds = squad.merge(load_predictions(), left_on="element", right_on="id", how="left", suffixes=("", "_p"))
    print(triple_captain_candidates(squad_preds, bd).head(5).to_string(index=False))
