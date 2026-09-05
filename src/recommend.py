"""Turn model predictions into concrete decisions for one FPL squad:
best starting XI, captain/vice, and transfer targets.

Uses REAL sell price (via squad_value.py's reconstruction of purchase price
+ FPL's half-profit-on-rise rule) for transfer budgeting, and folds in each
player's next-fixture difficulty (via fixtures_fdr.py) as context.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

from fpl_api import entry, entry_picks, live_points_by_element
from squad_value import squad_value_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POS_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
STARTERS = 11

# Players never suggested for transfer OUT, regardless of model score.
#
# Empty by default. This module is imported by a PUBLIC, multi-user Streamlit
# app where any visitor can view any manager's squad, so a name hardcoded here
# silently removes that player from EVERY visitor's transfer suggestions, not
# just the repo owner's. Pass `protected=` per call instead of editing this.
PROTECTED_PLAYERS: set[str] = set()


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "player_predictions.csv")


def load_squad(team_id: int, event: int) -> tuple[pd.DataFrame, dict]:
    picks = entry_picks(team_id, event)
    e = entry(team_id)
    squad = pd.DataFrame(picks["picks"]).rename(columns={"position": "squad_slot"})
    preds = load_predictions()
    squad = squad.merge(preds, left_on="element", right_on="id", how="left")

    meta = {
        "bank": picks["entry_history"]["bank"] / 10.0,
        "value": picks["entry_history"]["value"] / 10.0,
        "event": event,
        "team_name": e["name"],
    }
    return squad, meta


def live_squad_score(squad: pd.DataFrame, event: int) -> dict:
    """Actual accumulating points for this squad in `event`, using FPL's live
    endpoint (updates during matches -- goals/bonus tick the total up in
    near-real-time). NOT a prediction: this is what actually happened/is
    happening, for a gameweek whose deadline has passed. Captain multiplier
    from the picks themselves is respected (already reflects any auto-vice
    substitution FPL itself has applied if the captain didn't play).
    """
    live = live_points_by_element(event)
    s = squad.copy()
    s["live_points"] = s["element"].map(live).fillna(0).astype(int)
    starters = s[s["multiplier"] > 0]
    bench = s[s["multiplier"] == 0]
    return {
        "live_total": int((starters["live_points"] * starters["multiplier"]).sum()),
        "bench_points": int(bench["live_points"].sum()),
        "by_player": s[["web_name", "position", "multiplier", "live_points"]].sort_values(
            "live_points", ascending=False
        ),
    }


def best_starting_xi(squad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    squad = squad.sort_values("pred_points_adj", ascending=False)
    by_pos = {p: squad[squad.position == p].reset_index(drop=True) for p in POS_LIMITS}

    best = None
    best_pts = -1
    def_range = range(POS_LIMITS["DEF"][0], POS_LIMITS["DEF"][1] + 1)
    mid_range = range(POS_LIMITS["MID"][0], POS_LIMITS["MID"][1] + 1)
    fwd_range = range(POS_LIMITS["FWD"][0], POS_LIMITS["FWD"][1] + 1)

    for d, m, f in itertools.product(def_range, mid_range, fwd_range):
        if 1 + d + m + f != STARTERS:
            continue
        if d > len(by_pos["DEF"]) or m > len(by_pos["MID"]) or f > len(by_pos["FWD"]):
            continue
        chosen = pd.concat([
            by_pos["GKP"].head(1),
            by_pos["DEF"].head(d),
            by_pos["MID"].head(m),
            by_pos["FWD"].head(f),
        ])
        pts = chosen["pred_points_adj"].sum()
        if pts > best_pts:
            best_pts, best = pts, chosen

    bench = squad[~squad["element"].isin(best["element"])]
    bench = bench.sort_values("pred_points_adj", ascending=False)
    return best.sort_values("pred_points_adj", ascending=False), bench


def suggest_transfers(squad_with_value: pd.DataFrame, bank: float, top_n: int = 3,
                      protected: set[str] | None = None) -> pd.DataFrame:
    """protected: player web_names to never suggest selling. Defaults to the
    module-level PROTECTED_PLAYERS (empty), so a caller opts in per squad
    rather than the whole deployment inheriting one person's preference."""
    preds = load_predictions()
    preds = preds[preds["status_ok"]]
    owned_ids = set(squad_with_value["element"])
    suggestions = []

    keep = PROTECTED_PLAYERS if protected is None else protected
    candidate_rows = squad_with_value[~squad_with_value["web_name"].isin(keep)]
    for _, row in candidate_rows.sort_values("pred_points_adj").iterrows():
        pos = row["position"]
        budget = row["sell_price_m"] + bank  # real sellable budget, not market price
        candidates = preds[
            (preds["position"] == pos)
            & (~preds["id"].isin(owned_ids))
            & (preds["now_cost_m"] <= budget)
        ].sort_values("pred_points_adj", ascending=False).head(top_n)
        gain = candidates["pred_points_adj"].values[:1]
        if len(gain) and gain[0] > row["pred_points_adj"] + 0.5:
            for _, cand in candidates.iterrows():
                suggestions.append({
                    "out": row["web_name"], "out_pos": pos,
                    "out_sell_price": row["sell_price_m"],
                    "out_pred": round(row["pred_points_adj"], 2),
                    "in": cand["web_name"], "in_team": cand["name"], "in_cost": cand["now_cost_m"],
                    "in_pred": round(cand["pred_points_adj"], 2),
                    "in_price_flag": cand["price_flag"],
                    "pred_gain": round(cand["pred_points_adj"] - row["pred_points_adj"], 2),
                    "leftover_bank": round(budget - cand["now_cost_m"], 1),
                })
    return pd.DataFrame(suggestions).sort_values("pred_gain", ascending=False) if suggestions else pd.DataFrame()


def report(team_id: int, event: int):
    squad, meta = load_squad(team_id, event)
    sv = squad_value_report(team_id, squad)
    squad = squad.merge(
        sv[["element", "buy_price_m", "sell_price_m", "profit_m"]], on="element", how="left"
    )

    print(f"=== {meta['team_name']} - squad entering GW{event + 1} ===")
    print(f"Bank: GBP{meta['bank']:.1f}m | Market value: GBP{meta['value']:.1f}m | "
          f"True sellable value: GBP{sv['sell_price_m'].sum():.1f}m "
          f"(unrealised profit: GBP{sv['profit_m'].sum():.1f}m)\n")

    xi, bench = best_starting_xi(squad)
    show_cols = ["web_name", "position", "name", "now_cost_m", "pred_points", "pred_points_adj", "next_fixture", "next_fdr"]
    print("--- Model-optimal starting XI (fixture-adjusted predicted points) ---")
    print(xi[show_cols].to_string(index=False))
    print(f"\nPredicted XI total (fixture-adjusted): {xi['pred_points_adj'].sum():.1f} pts")

    print("\n--- Bench ---")
    print(bench[show_cols].to_string(index=False))

    cap = xi.iloc[0]
    vice = xi.iloc[1]
    print(f"\nSuggested Captain: {cap['web_name']} vs {cap['next_fixture']} (FDR {cap['next_fdr']:.0f}) - {cap['pred_points_adj']:.1f} pts adj")
    print(f"Suggested Vice:    {vice['web_name']} vs {vice['next_fixture']} (FDR {vice['next_fdr']:.0f}) - {vice['pred_points_adj']:.1f} pts adj")

    actual_cap = squad[squad["is_captain"] == True]
    if not actual_cap.empty:
        ac = actual_cap.iloc[0]
        print(f"\nYour actual captain: {ac['web_name']} ({ac['pred_points_adj']:.1f} pts adj, FDR {ac['next_fdr']:.0f})")

    price_alerts = squad[squad["price_flag"] != ""]
    if not price_alerts.empty:
        print("\n--- Price-change alerts (your squad, FPL's own transfer-momentum signal) ---")
        print(price_alerts[["web_name", "price_flag", "price_change_percent"]].to_string(index=False))

    print(f"\n--- Squad value (buy price -> real sell price) ---")
    print(squad[["web_name", "buy_price_m", "now_cost_m", "sell_price_m", "profit_m"]]
          .sort_values("profit_m", ascending=False).to_string(index=False))

    print(f"\n--- Transfer suggestions (upgrade > 0.5 pred pts, within real sell budget; "
          f"{', '.join(PROTECTED_PLAYERS) or 'none'} excluded from OUT list) ---")
    tx = suggest_transfers(squad, meta["bank"])
    if tx.empty:
        print("No clear upgrades found within budget - squad looks efficient per the model.")
    else:
        print(tx.head(10).to_string(index=False))


if __name__ == "__main__":
    team_id = int(sys.argv[1]) if len(sys.argv) > 1 else 8041052
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    report(team_id, event)
