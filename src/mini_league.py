"""Mini-league tracking: standings plus every manager's squad and league-wide
insights (most-captained, template/most-owned players, your differentials,
chip usage). Entirely public endpoints -- once a gameweek's deadline has
passed, anyone's picks for that gameweek are publicly readable, no auth.
"""
import concurrent.futures as cf
from pathlib import Path

import pandas as pd
import requests

from fpl_api import BASE, SESSION, bootstrap_static, entry_history, entry_picks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def league_standings(league_id: int) -> pd.DataFrame:
    """All pages of a classic league's standings."""
    rows = []
    page = 1
    while True:
        r = SESSION.get(f"{BASE}/leagues-classic/{league_id}/standings/",
                         params={"page_standings": page}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if page == 1:
            league_name = data["league"]["name"]
        rows.extend(data["standings"]["results"])
        if not data["standings"]["has_next"]:
            break
        page += 1
    df = pd.DataFrame(rows)
    df.attrs["league_name"] = league_name
    return df


def fetch_league_picks(entry_ids: list[int], event: int, workers: int = 12) -> dict[int, dict]:
    """entry_id -> raw picks response (picks list + entry_history), for every
    manager in the league, for one gameweek."""
    out = {}

    def _fetch(eid):
        try:
            return eid, entry_picks(eid, event)
        except requests.HTTPError:
            return eid, None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for eid, data in ex.map(_fetch, entry_ids):
            if data is not None:
                out[eid] = data
    return out


def build_league_squads(standings: pd.DataFrame, event: int) -> pd.DataFrame:
    """One row per (manager, player-in-their-squad), with player name/position
    /cost/captaincy, ready to aggregate into league-wide stats."""
    bs = bootstrap_static()
    elements = pd.DataFrame(bs["elements"])[["id", "web_name", "now_cost"]]
    positions = pd.DataFrame(bs["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "element_type", "singular_name_short": "position"}
    )
    elements = elements.merge(
        pd.DataFrame(bs["elements"])[["id", "element_type"]], on="id"
    ).merge(positions, on="element_type", how="left")

    picks_by_entry = fetch_league_picks(standings["entry"].tolist(), event)
    rows = []
    for _, mgr in standings.iterrows():
        data = picks_by_entry.get(mgr["entry"])
        if not data:
            continue
        for p in data["picks"]:
            rows.append({
                "entry": mgr["entry"],
                "entry_name": mgr["entry_name"],
                "player_name": mgr["player_name"],
                "rank": mgr["rank"],
                "element": p["element"],
                "is_captain": p["is_captain"],
                "is_vice_captain": p["is_vice_captain"],
                "multiplier": p["multiplier"],
                "active_chip": data.get("active_chip"),
                "event_transfers_cost": data["entry_history"]["event_transfers_cost"],
            })
    df = pd.DataFrame(rows)
    df = df.merge(elements[["id", "web_name", "now_cost", "position"]], left_on="element", right_on="id", how="left")
    df["cost_m"] = df["now_cost"] / 10.0
    return df


def league_insights(league_squads: pd.DataFrame, my_entry_id: int) -> dict:
    n_managers = league_squads["entry"].nunique()

    ownership = (
        league_squads.groupby(["element", "web_name", "position"])["entry"]
        .nunique().rename("owned_by_n").reset_index()
        .sort_values("owned_by_n", ascending=False)
    )
    ownership["owned_pct"] = (ownership["owned_by_n"] / n_managers * 100).round(0)

    captains = (
        league_squads[league_squads["is_captain"]]
        .groupby(["element", "web_name"])["entry"]
        .nunique().rename("captained_by_n").reset_index()
        .sort_values("captained_by_n", ascending=False)
    )

    my_players = set(league_squads[league_squads["entry"] == my_entry_id]["element"])
    differentials = ownership[
        (ownership["element"].isin(my_players)) & (ownership["owned_by_n"] <= max(1, n_managers // 5))
    ].sort_values("owned_by_n")

    chips = (
        league_squads[["entry", "entry_name", "player_name", "active_chip"]]
        .drop_duplicates()
        .query("active_chip.notna()", engine="python")
    )

    template = ownership[ownership["owned_pct"] >= 50]

    return {
        "n_managers": n_managers,
        "ownership": ownership,
        "captains": captains,
        "differentials": differentials,
        "chips_active_this_gw": chips,
        "template": template,
    }


def league_rank_progression(standings: pd.DataFrame, workers: int = 12) -> pd.DataFrame:
    """GW-by-GW cumulative total points for every manager in the league.
    Returns a DataFrame indexed by gameweek, one column per entry_name."""
    def _fetch(row):
        try:
            hist = entry_history(row["entry"])["current"]
            return row["entry_name"], {h["event"]: h["total_points"] for h in hist}
        except Exception:
            return row["entry_name"], {}

    series = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, points_by_gw in ex.map(_fetch, [r for _, r in standings.iterrows()]):
            series[name] = points_by_gw

    df = pd.DataFrame(series)
    df.index.name = "event"
    return df.sort_index()


if __name__ == "__main__":
    import sys
    league_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1766517
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    my_entry = int(sys.argv[3]) if len(sys.argv) > 3 else 8041052

    standings = league_standings(league_id)
    print(f"=== {standings.attrs['league_name']} ===")
    print(standings[["rank", "entry_name", "player_name", "event_total", "total"]].to_string(index=False))

    squads = build_league_squads(standings, event)
    insights = league_insights(squads, my_entry)
    print(f"\n--- Most owned in league ({insights['n_managers']} managers) ---")
    print(insights["ownership"].head(15).to_string(index=False))
    print("\n--- Captaincy choices ---")
    print(insights["captains"].to_string(index=False))
    print("\n--- Your differentials (low ownership in this league) ---")
    print(insights["differentials"].to_string(index=False))
    if not insights["chips_active_this_gw"].empty:
        print("\n--- Chips played this gameweek ---")
        print(insights["chips_active_this_gw"].to_string(index=False))
