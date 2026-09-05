"""Download multi-season gameweek-level FPL data (vaastav/Fantasy-Premier-League archive)
plus the live current-season bootstrap data, and cache everything under data/raw/.
"""
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


if __name__ == "__main__":
    build_history()
