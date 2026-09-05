"""Thin wrapper around the official (public, unauthenticated) Fantasy Premier League API."""
import json
import time
from pathlib import Path

import requests

BASE = "https://fantasy.premierleague.com/api"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fpl-assistant/1.0"})


def _get(path: str) -> dict:
    r = SESSION.get(f"{BASE}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _cached_get(path: str, cache_name: str, max_age_s: int = 3600) -> dict:
    cache_path = DATA_DIR / cache_name
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < max_age_s:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    data = _get(path)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def bootstrap_static(max_age_s: int = 1800) -> dict:
    """All players, teams, gameweeks (events), positions. The core reference dataset."""
    return _cached_get("bootstrap-static/", "bootstrap-static.json", max_age_s)


def fixtures(max_age_s: int = 1800) -> list:
    return _cached_get("fixtures/", "fixtures.json", max_age_s)


def entry(team_id: int) -> dict:
    return _get(f"entry/{team_id}/")


def entry_history(team_id: int) -> dict:
    return _get(f"entry/{team_id}/history/")


def entry_picks(team_id: int, event: int) -> dict:
    return _get(f"entry/{team_id}/event/{event}/picks/")


def entry_transfers(team_id: int) -> list:
    return _get(f"entry/{team_id}/transfers/")


def player_summary(player_id: int) -> dict:
    """Per-player: history (this season, gw by gw) and history_past (prior seasons)."""
    return _get(f"element-summary/{player_id}/")


def current_event(bootstrap: dict | None = None) -> int:
    bs = bootstrap or bootstrap_static()
    for e in bs["events"]:
        if e["is_current"]:
            return e["id"]
    for e in bs["events"]:
        if e["is_next"]:
            return e["id"] - 1
    return max(e["id"] for e in bs["events"] if e["finished"])
