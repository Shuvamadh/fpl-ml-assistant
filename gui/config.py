"""Tiny local config: remembers the last team ID / event used so the GUI
doesn't need retyping every launch."""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "gui_config.json"

DEFAULTS = {"team_id": 8041052, "event": 2, "league_id": 1766517, "theme": "Graphite"}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)


def save_config(team_id: int, event: int, league_id: int, theme: str | None = None):
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    payload = {"team_id": team_id, "event": event, "league_id": league_id}
    if theme is None:
        try:
            payload["theme"] = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("theme", DEFAULTS["theme"])
        except Exception:
            payload["theme"] = DEFAULTS["theme"]
    else:
        payload["theme"] = theme
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload), encoding="utf-8")
