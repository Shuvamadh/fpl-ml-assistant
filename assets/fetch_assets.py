"""Download official Premier League / FPL image assets for local GUI use.

Cached on disk under assets/ -- every fetch is skipped if the file already
exists, so this is cheap to re-run and safe to call at app startup.

Assets pulled:
    assets/badges/t{code}.png    club crest, 70px      (20 files)
    assets/shirts/shirt_{code}.png   kit thumbnail     (20 files)
    assets/players/p{code}.png   player mugshot 110x140 (on demand)

These are Premier League copyright assets fetched from the same public CDN the
official FPL site uses. Fine for a personal local dashboard; don't redistribute.
"""
import concurrent.futures as cf
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
BADGES = ASSETS / "badges"
SHIRTS = ASSETS / "shirts"
PLAYERS = ASSETS / "players"

BADGE_URL = "https://resources.premierleague.com/premierleague/badges/70/t{code}.png"
SHIRT_URL = "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}-66.png"
SHIRT_GK_URL = "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}_1-66.png"
PHOTO_URL = "https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"

TIMEOUT = 20


def _get(url: str, dest: Path) -> bool:
    """Fetch url -> dest. Returns True if a new file was written."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200 or not r.content:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def _bootstrap() -> dict:
    return json.loads((ROOT / "data" / "bootstrap-static.json").read_text(encoding="utf-8"))


def fetch_club_assets(bs: dict | None = None, workers: int = 8) -> int:
    """Crests + outfield/GK kits for all 20 clubs. ~40 small files."""
    bs = bs or _bootstrap()
    jobs = []
    for t in bs["teams"]:
        c = t["code"]
        jobs.append((BADGE_URL.format(code=c), BADGES / f"t{c}.png"))
        jobs.append((SHIRT_URL.format(code=c), SHIRTS / f"shirt_{c}.png"))
        jobs.append((SHIRT_GK_URL.format(code=c), SHIRTS / f"shirt_{c}_gk.png"))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(lambda j: _get(*j), jobs))


def fetch_player_photos(codes, workers: int = 8) -> int:
    """Mugshots for the given player `code` values (element['code'])."""
    jobs = [(PHOTO_URL.format(code=c), PLAYERS / f"p{c}.png") for c in codes]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(lambda j: _get(*j), jobs))


def badge_path(team_code: int) -> Path | None:
    p = BADGES / f"t{team_code}.png"
    return p if p.exists() else None


def shirt_path(team_code: int, is_gk: bool = False) -> Path | None:
    p = SHIRTS / (f"shirt_{team_code}_gk.png" if is_gk else f"shirt_{team_code}.png")
    return p if p.exists() else None


def photo_path(player_code: int) -> Path | None:
    p = PLAYERS / f"p{player_code}.png"
    return p if p.exists() else None


def ensure_photo(player_code: int) -> Path | None:
    """Lazy single-player fetch, for rows the user actually looks at."""
    dest = PLAYERS / f"p{player_code}.png"
    if not dest.exists():
        _get(PHOTO_URL.format(code=player_code), dest)
    return dest if dest.exists() else None


def main(top_n: int = 150) -> None:
    bs = _bootstrap()
    n = fetch_club_assets(bs)
    print(f"club assets: {n} new ({len(bs['teams'])} clubs)")

    by_id = {e["id"]: e for e in bs["elements"]}
    wanted: list[int] = []

    # my squad first
    picks = ROOT / "data" / "entry_8041052_picks_gw2.json"
    if picks.exists():
        data = json.loads(picks.read_text(encoding="utf-8"))
        for p in data.get("picks", []):
            e = by_id.get(p["element"])
            if e:
                wanted.append(e["code"])
        print(f"squad players: {len(wanted)}")

    # then the most-owned/most-relevant players, so common rows are warm
    ranked = sorted(
        bs["elements"],
        key=lambda e: float(e.get("selected_by_percent") or 0),
        reverse=True,
    )[:top_n]
    wanted += [e["code"] for e in ranked]

    wanted = list(dict.fromkeys(wanted))
    n = fetch_player_photos(wanted)
    print(f"player photos: {n} new, {len(wanted)} requested")
    print(f"cache: {sum(1 for _ in PLAYERS.glob('*.png'))} photos on disk")


if __name__ == "__main__":
    main()
