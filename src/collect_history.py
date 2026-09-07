"""Download multi-season gameweek-level FPL data (vaastav/Fantasy-Premier-League archive)
plus the live current-season bootstrap data, and cache everything under data/raw/.
"""
import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def fetch_season_gws(season: str) -> pd.DataFrame:
    cache = RAW_DIR / f"gws_{season}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    url = f"{BASE}/{season}/gws/merged_gw.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    cache.write_bytes(r.content)
    # some seasons are latin-1 encoded (accented player names)
    try:
        return pd.read_csv(cache)
    except UnicodeDecodeError:
        df = pd.read_csv(cache, encoding="latin-1")
        df.to_csv(cache, index=False)
        return df


def build_history() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        try:
            df = fetch_season_gws(season)
        except requests.HTTPError as e:
            print(f"skip {season}: {e}")
            continue
        df["season"] = season
        frames.append(df)
        print(f"{season}: {len(df)} rows")
    full = pd.concat(frames, ignore_index=True, sort=False)
    full.to_csv(DATA_DIR / "history_gws.csv", index=False)
    print(f"total: {len(full)} rows -> data/history_gws.csv")
    return full


def build_team_strength() -> pd.DataFrame:
    """Per-season FPL team strength ratings (teams.csv in the archive, present
    2019-20 onward). Consumed by src/components.py as a static team-strength
    prior for the thin clean-sheet / goals-conceded models."""
    rows = []
    for season in SEASONS:
        url = f"{BASE}/{season}/teams.csv"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"teams.csv skip {season}: HTTP {r.status_code}")
            continue
        (RAW_DIR / f"teams_{season}.csv").write_bytes(r.content)
        d = pd.read_csv(RAW_DIR / f"teams_{season}.csv")
        d["season"] = season
        cols = ["season", "id", "name", "short_name", "strength",
                "strength_overall_home", "strength_overall_away",
                "strength_attack_home", "strength_attack_away",
                "strength_defence_home", "strength_defence_away"]
        rows.append(d[[c for c in cols if c in d.columns]])
        print(f"teams {season}: {len(d)} teams")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA_DIR / "team_strength.csv", index=False)
    print(f"-> data/team_strength.csv ({len(out)} team-seasons)")
    return out


def build_player_codes() -> pd.DataFrame:
    """(season, element id) -> stable FPL player ``code``, from players_raw.csv.
    The archive reassigns element ids every season; ``code`` is stable, so
    this is what cross-season career features must key on."""
    rows = []
    for season in SEASONS:
        url = f"{BASE}/{season}/players_raw.csv"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"players_raw skip {season}: HTTP {r.status_code}")
            continue
        try:
            d = pd.read_csv(io.BytesIO(r.content))
        except UnicodeDecodeError:
            d = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
        if not {"id", "code"}.issubset(d.columns):
            continue
        d = d[["id", "code"]].copy()
        d["season"] = season
        rows.append(d.rename(columns={"id": "element"}))
        print(f"player_codes {season}: {len(d)} players")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA_DIR / "player_codes.csv", index=False)
    print(f"-> data/player_codes.csv ({len(out)} player-seasons)")
    return out


if __name__ == "__main__":
    build_history()
    build_team_strength()
    build_player_codes()
