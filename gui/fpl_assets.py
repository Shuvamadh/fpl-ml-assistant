"""Resolve FPL image assets (club crests, kits, player mugshots) to on-disk
files, and hand them back as QPixmaps.

The prediction/squad frames carry FPL *ids* (team_id 1-20, element id), but
the CDN filenames are keyed by *codes* (team code 3/7/91..., element code) --
so everything here goes through a bootstrap-derived id -> code map.

Assets are fetched by assets/fetch_assets.py; anything missing degrades to a
club kit, and then to None, so a missing image never breaks a render.
"""
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
sys.path.insert(0, str(ASSETS)) if str(ASSETS) not in sys.path else None

_team_code: dict[int, int] = {}
_player_code: dict[int, int] = {}
_player_team: dict[int, int] = {}
_loaded = False
_cache: dict[tuple, QPixmap] = {}


def _load_maps() -> None:
    global _loaded
    if _loaded:
        return
    path = ROOT / "data" / "bootstrap-static.json"
    if not path.exists():
        _loaded = True
        return
    try:
        bs = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _loaded = True
        return
    for t in bs.get("teams", []):
        _team_code[t["id"]] = t["code"]
    for e in bs.get("elements", []):
        _player_code[e["id"]] = e["code"]
        _player_team[e["id"]] = e["team"]
    _loaded = True


def team_code(team_id: int) -> int | None:
    _load_maps()
    return _team_code.get(int(team_id)) if team_id == team_id else None


def _pix(path: Path | None, height: int) -> QPixmap | None:
    """Load and scale, cached by (path, height)."""
    if path is None or not path.exists():
        return None
    key = (str(path), height)
    if key in _cache:
        return _cache[key]
    pm = QPixmap(str(path))
    if pm.isNull():
        return None
    pm = pm.scaledToHeight(height, Qt.SmoothTransformation)
    _cache[key] = pm
    return pm


def badge(team_id: int, height: int = 28) -> QPixmap | None:
    code = team_code(team_id)
    if code is None:
        return None
    return _pix(ASSETS / "badges" / f"t{code}.png", height)


def shirt(team_id: int, is_gk: bool = False, height: int = 56) -> QPixmap | None:
    code = team_code(team_id)
    if code is None:
        return None
    name = f"shirt_{code}_gk.png" if is_gk else f"shirt_{code}.png"
    pm = _pix(ASSETS / "shirts" / name, height)
    if pm is None and is_gk:  # some clubs have no separate GK kit cached
        pm = _pix(ASSETS / "shirts" / f"shirt_{code}.png", height)
    return pm


def photo(element_id: int, height: int = 60) -> QPixmap | None:
    """Player mugshot. Returns None when the CDN has no photo for them --
    recent signings genuinely have none, so callers must fall back."""
    _load_maps()
    code = _player_code.get(int(element_id))
    if code is None:
        return None
    return _pix(ASSETS / "players" / f"p{code}.png", height)


def player_image(element_id: int, team_id: int, is_gk: bool = False,
                 height: int = 56, prefer_photo: bool = True) -> QPixmap | None:
    """Best available image: mugshot if we have one, else the club kit."""
    if prefer_photo:
        pm = photo(element_id, height)
        if pm is not None:
            return pm
    return shirt(team_id, is_gk=is_gk, height=height)
