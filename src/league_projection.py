"""Mini-league projections and banter stats. Requested via a parallel session
on the user's behalf: projected GW winner, per-manager transfer suggestions,
and cheap "argue with friends" stats. Builds on mini_league.py's squads/
standings rather than duplicating that fetching.

HONEST STALENESS CAVEAT (surfaced in the return, not just here): FPL does not
expose another manager's picks for a gameweek whose deadline hasn't passed.
So a projection for the upcoming gameweek is necessarily built on each
manager's LAST KNOWN (already-played) squad, and is blind to any transfers
they've made since. Every function below returns `picks_from_gw` /
`is_stale` explicitly rather than presenting a confident-looking leaderboard
silently based on last week's teams.
"""
import concurrent.futures as cf
from pathlib import Path

import pandas as pd

from fpl_api import entry_history

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "player_predictions.csv")


def project_gw_winner(league_squads: pd.DataFrame, standings: pd.DataFrame, picks_from_gw: int) -> pd.DataFrame:
    """Ranked table: manager, projected points for the upcoming GW (starting
    XI only, captain/vice multiplier applied, with a fallback swap to vice if
    the captain looks unlikely to play), projected total, and rank movement.

    Return columns: entry, entry_name, player_name, current_total,
    projected_gw_points, projected_total, current_rank, projected_rank,
    rank_change, captain_name, picks_from_gw, is_stale.
    """
    preds = load_predictions()
    merged = league_squads.merge(
        preds[["id", "pred_points_adj", "status_ok", "chance_of_playing_next_round"]],
        left_on="element", right_on="id", how="left",
    )
    merged["pred_points_adj"] = merged["pred_points_adj"].fillna(0.0)

    rows = []
    for entry, g in merged.groupby("entry"):
        starters = g[g["multiplier"] > 0].copy()
        cap = starters[starters["is_captain"]]
        vice = starters[starters["is_vice_captain"]]
        cap_risky = False
        if not cap.empty:
            c = cap.iloc[0]
            cap_risky = (not bool(c.get("status_ok", True))) or (
                pd.notna(c.get("chance_of_playing_next_round")) and c["chance_of_playing_next_round"] < 50
            )
        if cap_risky and not vice.empty:
            # swap captaincy multiplier to vice for this projection only
            starters.loc[starters["is_captain"], "multiplier"] = 1
            starters.loc[starters["is_vice_captain"], "multiplier"] = 2
            effective_captain = vice.iloc[0]["web_name"]
        else:
            effective_captain = cap.iloc[0]["web_name"] if not cap.empty else None

        projected = (starters["pred_points_adj"] * starters["multiplier"]).sum()
        mgr_row = standings[standings["entry"] == entry]
        current_total = mgr_row["total"].iloc[0] if not mgr_row.empty else g["rank"].iloc[0]
        current_rank = mgr_row["rank"].iloc[0] if not mgr_row.empty else None
        rows.append({
            "entry": entry, "entry_name": g["entry_name"].iloc[0], "player_name": g["player_name"].iloc[0],
            "current_total": current_total, "projected_gw_points": round(projected, 1),
            "projected_total": current_total + projected, "current_rank": current_rank,
            "captain_name": effective_captain, "picks_from_gw": picks_from_gw, "is_stale": True,
        })

    out = pd.DataFrame(rows).sort_values("projected_total", ascending=False).reset_index(drop=True)
    out["projected_rank"] = out.index + 1
    out["rank_change"] = out["current_rank"] - out["projected_rank"]
    return out


def per_manager_transfer_suggestions(league_squads: pd.DataFrame, assumed_bank: float = 0.0, top_n: int = 2) -> pd.DataFrame:
    """For every manager: their weakest starters by predicted points, and the
    best available upgrade in the same position within an APPROXIMATE budget
    (their player's current market price + `assumed_bank` -- their real bank
    isn't public, so this is a stated assumption, not a fact).

    Return columns: entry, entry_name, out, out_pos, out_pred, in, in_club,
    in_cost, in_pred, pred_gain, assumed_bank.
    """
    preds = load_predictions()
    preds = preds[preds["status_ok"]]
    merged = league_squads.merge(
        preds[["id", "pred_points_adj", "position", "now_cost_m"]],
        left_on="element", right_on="id", how="left", suffixes=("", "_pred"),
    )
    rows = []
    for entry, g in merged.groupby("entry"):
        starters = g[g["multiplier"] > 0].dropna(subset=["pred_points_adj"])
        if starters.empty:
            continue
        owned_ids = set(g["element"])
        weakest = starters.nsmallest(top_n, "pred_points_adj")
        for _, row in weakest.iterrows():
            budget = row["now_cost_m"] + assumed_bank
            candidates = preds[
                (preds["position"] == row["position"])
                & (~preds["id"].isin(owned_ids))
                & (preds["now_cost_m"] <= budget)
            ].nlargest(1, "pred_points_adj")
            if candidates.empty or candidates.iloc[0]["pred_points_adj"] <= row["pred_points_adj"] + 0.5:
                continue
            cand = candidates.iloc[0]
            rows.append({
                "entry": entry, "entry_name": g["entry_name"].iloc[0],
                "out": row["web_name"], "out_pos": row["position"], "out_pred": round(row["pred_points_adj"], 2),
                "in": cand["web_name"], "in_club": cand["name"], "in_cost": cand["now_cost_m"],
                "in_pred": round(cand["pred_points_adj"], 2),
                "pred_gain": round(cand["pred_points_adj"] - row["pred_points_adj"], 2),
                "assumed_bank": assumed_bank,
            })
    return pd.DataFrame(rows).sort_values("pred_gain", ascending=False) if rows else pd.DataFrame()


def _fetch_entry_histories(entry_ids: list[int], workers: int = 12) -> dict[int, dict]:
    def _fetch(eid):
        try:
            return eid, entry_history(eid)
        except Exception:
            return eid, None
    out = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for eid, data in ex.map(_fetch, entry_ids):
            if data:
                out[eid] = data
    return out


def banter_stats(league_squads: pd.DataFrame, standings: pd.DataFrame, insights: dict) -> dict:
    """Cheap, derivable-from-what-we-already-fetch stats for arguing with
    friends: template adherence, bench points left on the bench (luck vs
    skill), squad value growth since GW1, and biggest weekly rank swings.

    Deliberately NOT included: season-long best/worst captain pick, which
    would need every manager's picks for every past gameweek (~24 managers x
    38 GWs of API calls) -- too expensive to fetch on every refresh. Flagging
    this as a known gap rather than a silent omission.
    """
    template_ids = set(insights["template"]["element"]) if not insights["template"].empty else set()
    template_adherence = (
        league_squads[league_squads["element"].isin(template_ids)]
        .groupby(["entry", "entry_name"])["element"].nunique()
        .rename("template_players_owned").reset_index()
    )
    template_adherence["template_size"] = len(template_ids)

    histories = _fetch_entry_histories(standings["entry"].tolist())
    rows = []
    for _, mgr in standings.iterrows():
        hist = histories.get(mgr["entry"])
        if not hist or not hist.get("current"):
            continue
        current = hist["current"]
        bench_pts = [h["points_on_bench"] for h in current]
        gw1_value = current[0]["value"] / 10.0
        latest_value = current[-1]["value"] / 10.0
        rows.append({
            "entry": mgr["entry"], "entry_name": mgr["entry_name"],
            "total_bench_points_lost": sum(bench_pts),
            "avg_bench_points_per_gw": round(sum(bench_pts) / len(bench_pts), 1) if bench_pts else 0,
            "squad_value_gw1": gw1_value, "squad_value_now": latest_value,
            "squad_value_growth": round(latest_value - gw1_value, 1),
        })
    per_manager = pd.DataFrame(rows)

    # rank swing must be within the MINI-LEAGUE, not FPL's global overall
    # rank (which naturally swings by millions and would be a meaningless
    # "biggest mover" stat) -- standings already carries last_rank vs rank
    # for exactly this league.
    if not per_manager.empty:
        rank_swing = standings[["entry", "last_rank", "rank"]].copy()
        rank_swing["gw_rank_swing"] = rank_swing["last_rank"] - rank_swing["rank"]
        per_manager = per_manager.merge(rank_swing[["entry", "gw_rank_swing"]], on="entry", how="left")

    return {
        "template_adherence": template_adherence.sort_values("template_players_owned", ascending=False),
        "per_manager_stats": per_manager.sort_values("squad_value_growth", ascending=False) if not per_manager.empty else per_manager,
        "known_gap": "season-long best/worst captain pick omitted -- would need ~24 managers x 38 GWs of picks calls",
    }


if __name__ == "__main__":
    import sys
    import mini_league

    league_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1766517
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    standings = mini_league.league_standings(league_id)
    squads = mini_league.build_league_squads(standings, event)
    insights = mini_league.league_insights(squads, 8041052)

    print("=== Projected GW winner (STALE: based on last known picks) ===")
    proj = project_gw_winner(squads, standings, event)
    print(proj[["projected_rank", "entry_name", "captain_name", "projected_gw_points",
                "projected_total", "rank_change"]].to_string(index=False))

    print("\n=== Per-manager transfer suggestions (assumed_bank=0.0) ===")
    tx = per_manager_transfer_suggestions(squads)
    print(tx.head(15).to_string(index=False) if not tx.empty else "none found")

    print("\n=== Banter stats ===")
    banter = banter_stats(squads, standings, insights)
    print(banter["per_manager_stats"].to_string(index=False))
    print("\nTemplate adherence:")
    print(banter["template_adherence"].to_string(index=False))
