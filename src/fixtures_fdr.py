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


def upcoming_fixtures_by_team(n: int = 5) -> pd.DataFrame:
    """One row per (team, upcoming fixture), earliest first, capped at n per team."""
    fx = fixtures()
    names = team_id_to_name()
    rows = []
    for f in fx:
        if f["finished"] or f["event"] is None:
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


def next_fixture_summary(n: int = 5) -> pd.DataFrame:
    """One row per team: next fixture string, its difficulty, and mean
    difficulty over the next n fixtures (lower = easier run)."""
    df = upcoming_fixtures_by_team(n)
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
