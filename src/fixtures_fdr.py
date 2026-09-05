"""Fixture difficulty helpers: for any team, the next N fixtures (opponent,
venue, FPL's own 1-5 difficulty rating). Used to (a) show each squad player's
next fixture in the recommender, and (b) feed a fixture-swing signal into
transfer/captaincy calls -- a player in great form facing a 5-difficulty away
game is a different bet than the same player at home to a 2.
"""
from pathlib import Path

import pandas as pd

from fpl_api import bootstrap_static, fixtures

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def team_id_to_name(bs: dict | None = None) -> dict[int, str]:
    bs = bs or bootstrap_static()
    return {t["id"]: t["short_name"] for t in bs["teams"]}


def last_finished_event(bs: dict | None = None) -> int:
    """Highest gameweek that is fully complete. During a live GW this is the
    PREVIOUS gameweek -- the boundary at which every player in the league has
    exactly the same amount of history behind them."""
    bs = bs or bootstrap_static()
    done = [e["id"] for e in bs["events"] if e.get("finished")]
    return max(done) if done else 0


def target_event(bs: dict | None = None) -> int:
    """The gameweek predictions are FOR: the one in progress if a gameweek is
    live, otherwise the next one up. Every player is scored against this same
    gameweek regardless of whether their own club has already played it."""
    return last_finished_event(bs) + 1


def upcoming_fixtures_by_team(n: int = 5, from_event: int | None = None) -> pd.DataFrame:
    """One row per (team, upcoming fixture), earliest first, capped at n per team.

    from_event anchors every team to the same gameweek. Without it this used
    `finished` to decide what counts as "upcoming", which breaks mid-gameweek:
    a club that has already played GW3 has its GW3 fixture dropped and rolls
    forward to GW4, while every other club still resolves to GW3. Predictions
    for the two groups then answer different questions but land in one ranked
    table. Anchoring on the event number keeps a played-but-current fixture in
    scope so all 20 clubs line up on the same gameweek.
    """
    fx = fixtures()
    names = team_id_to_name()
    rows = []
    for f in fx:
        if f["event"] is None:
            continue
        if from_event is not None:
            if f["event"] < from_event:
                continue
        elif f["finished"]:
            continue
        rows.append({
            "team": f["team_h"], "opponent": f["team_a"], "is_home": True,
            "difficulty": f["team_h_difficulty"], "event": f["event"],
            "kickoff": f["kickoff_time"],
        })
        rows.append({
            "team": f["team_a"], "opponent": f["team_h"], "is_home": False,
            "difficulty": f["team_a_difficulty"], "event": f["event"],
            "kickoff": f["kickoff_time"],
        })
    df = pd.DataFrame(rows).sort_values(["team", "event"])
    df["opponent_name"] = df["opponent"].map(names)
    df["team_name"] = df["team"].map(names)
    df["fixture_str"] = df["opponent_name"] + df["is_home"].map({True: " (H)", False: " (A)"})
    return df.groupby("team").head(n).reset_index(drop=True)


def next_fixture_summary(n: int = 5, from_event: int | None = None) -> pd.DataFrame:
    """One row per team: next fixture string, its difficulty, and mean
    difficulty over the next n fixtures (lower = easier run)."""
    df = upcoming_fixtures_by_team(n, from_event=from_event)
    if df.empty:
        return df
    first = df.sort_values(["team", "event"]).groupby("team").first().reset_index()
    mean_diff = df.groupby("team")["difficulty"].mean().rename("fdr_next_n_mean")
    out = first.merge(mean_diff, on="team")
    return out[["team", "team_name", "fixture_str", "difficulty", "fdr_next_n_mean", "is_home"]].rename(
        columns={"difficulty": "next_fdr", "fixture_str": "next_fixture"}
    )


def fixture_difficulty_matrix(n: int = 5) -> pd.DataFrame:
    """rows = team short name, cols = 'GW+1'..'GW+n', values = FDR (1-5)."""
    df = upcoming_fixtures_by_team(n)
    if df.empty:
        return df
    df = df.sort_values(["team", "event"])
    df["slot"] = df.groupby("team").cumcount() + 1
    pivot = df.pivot(index="team_name", columns="slot", values="difficulty")
    pivot.columns = [f"GW+{c}" for c in pivot.columns]
    mean_fdr = df.groupby("team_name")["difficulty"].mean()
    pivot = pivot.loc[mean_fdr.sort_values().index]
    return pivot


if __name__ == "__main__":
    print(next_fixture_summary().sort_values("fdr_next_n_mean").to_string(index=False))
