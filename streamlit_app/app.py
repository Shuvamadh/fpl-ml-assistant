"""FPL ML Assistant -- shareable Streamlit version.

Run with:  streamlit run streamlit_app/app.py

Runs entirely server-side, so every visitor gets the same live data pull.
Friends don't need Python, PySide6 or Ollama -- just a browser.

Two things this version fixes structurally:

1. CHARTS ARE NO LONGER DUPLICATED. Every figure comes from src/charts_core.py,
   which the desktop GUI also uses. Previously the 13 charts lived in
   gui/charts.py bound to a Qt canvas, so this app could not import them: two
   were hand-rewritten here and eleven were simply missing. Adding a chart to
   charts_core now lands in both frontends at once.

2. PREDICTION HORIZONS ARE NEVER MIXED. predict.score_players() defaults to
   as_of="frozen", scoring every player from the last completed gameweek so no
   club gets a head start mid-gameweek. Each row carries predicts_gw, and the
   UI states which gameweek it is showing rather than leaving it implied.

Key feature beyond the desktop app: the "Viewing as" picker lets anyone in the
mini-league select their own name and see the whole app built around their
squad -- one shared URL, everyone's own recommendations.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a cloud host; must precede pyplot
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
# gui/league_extras.py is deliberately Qt-free (plain DataFrames in/out) so it
# is shared rather than reimplemented. The REST of gui/ imports PySide6, which
# this deployment does not install, so only this directory goes on the path --
# never gui/data_bridge.py, and never gui/charts.py (now a thin Qt wrapper;
# src/charts_core.py is the Qt-free half and that is what we import).
GUI_DIR = ROOT / "gui"
ASSETS_DIR = ROOT / "assets"
for _d in (SRC_DIR, GUI_DIR, ASSETS_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import lightgbm as lgb

import charts_core
import chips
import fetch_assets
import fpl_api
import league_extras
import league_projection
import llm_assist
import mini_league
import nothing_ui as ui
import predict
import recommend
import squad_value as squad_value_mod
from features import FEATURE_COLS
from fixtures_fdr import fixture_difficulty_matrix

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DEFAULT_LEAGUE_ID = 1766517
DEFAULT_TEAM_ID = 8041052

_ICON = ROOT / "assets" / "fpl.ico"
st.set_page_config(
    page_title="FPL ML Assistant",
    page_icon=str(_ICON) if _ICON.exists() else "\u26bd",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================== caching ====

@st.cache_resource(show_spinner=False)
def ai_available() -> bool:
    """Ollama reachability. Cached as a resource because it was previously
    re-probed on EVERY rerun -- a 3s timeout on every widget interaction if the
    connection is dropped rather than refused."""
    try:
        return llm_assist.is_available()
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def get_active_picks_event() -> tuple[int, bool]:
    """(event_id, is_live). bootstrap's is_current event locks in the moment
    its deadline passes, so once a gameweek kicks off its picks are the real
    current squad. Probes with the default team since deadline timing is
    identical for everyone."""
    bs = fpl_api.bootstrap_static(max_age_s=120)
    current_id = fpl_api.current_event(bs)
    meta = next((e for e in bs["events"] if e["id"] == current_id), {})
    try:
        fpl_api.entry_picks(DEFAULT_TEAM_ID, current_id)
        return current_id, not meta.get("finished", True)
    except Exception:
        prev = max(current_id - 1, 1)
        prev_meta = next((e for e in bs["events"] if e["id"] == prev), {})
        return prev, not prev_meta.get("finished", True)


@st.cache_data(ttl=1800, show_spinner="Scoring ~650 players (live data pull)...")
def get_predictions(as_of: str) -> pd.DataFrame:
    df = predict.score_players(as_of=as_of)
    # recommend.py / chips.py / league_projection.py read predictions back off
    # this file rather than taking a DataFrame (the same contract the desktop
    # data_bridge relies on), so it must actually be written -- otherwise every
    # squad/transfer/chip call below fails on a fresh deploy with no file on
    # disk.
    predict.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(predict.DATA_DIR / "player_predictions.csv", index=False)
    return df


@st.cache_data(ttl=1800, show_spinner="Loading league standings...")
def get_standings(league_id: int) -> pd.DataFrame:
    return mini_league.league_standings(league_id)


@st.cache_data(ttl=1800, show_spinner="Pulling every manager's squad...")
def get_league_squads(league_id: int, event: int) -> pd.DataFrame:
    return mini_league.build_league_squads(get_standings(league_id), event)


@st.cache_data(ttl=1800, show_spinner=False)
def get_league_insights(league_id: int, event: int, my_entry_id: int) -> dict:
    return mini_league.league_insights(get_league_squads(league_id, event), my_entry_id)


@st.cache_data(ttl=600, show_spinner="Loading squad...")
def get_squad(team_id: int, event: int):
    squad, meta = recommend.load_squad(team_id, event)
    sv = squad_value_mod.squad_value_report(team_id, squad)
    squad = squad.merge(
        sv[["element", "buy_price_m", "sell_price_m", "profit_m"]], on="element", how="left"
    )
    return squad, meta


@st.cache_data(ttl=90, show_spinner=False)
def get_live_score(_squad: pd.DataFrame, event: int, team_id: int) -> dict:
    """Short TTL: this genuinely changes during matches. _squad is
    underscore-prefixed so Streamlit skips hashing a 15-row frame on every
    rerun; team_id + event are the real cache key."""
    return recommend.live_squad_score(_squad, event)


@st.cache_data(ttl=1800, show_spinner="Pulling GW-by-GW history for every manager...")
def get_manager_hist(league_id: int) -> pd.DataFrame:
    """One row per (manager, gameweek). Same shape as the desktop
    data_bridge._manager_gw_history, reimplemented rather than imported since
    that module pulls in PySide6 at import time."""
    import concurrent.futures as cf

    standings = get_standings(league_id)
    rows = []

    def _fetch(row):
        try:
            return row["entry_name"], fpl_api.entry_history(row["entry"])["current"]
        except Exception:
            return row["entry_name"], []

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for name, hist in ex.map(_fetch, [r for _, r in standings.iterrows()]):
            for h in hist:
                rows.append({**h, "entry_name": name})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fixture_matrix(n: int = 5) -> pd.DataFrame:
    try:
        return fixture_difficulty_matrix(n)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_model_mae() -> float:
    """Measured backtest MAE, for honest uncertainty bands. Falls back to the
    documented figure when backtest_results.csv is not present."""
    try:
        return float(pd.read_csv(DATA_DIR / "backtest_results.csv")["mae_model"].mean())
    except Exception:
        return 0.96


@st.cache_resource(show_spinner=False)
def get_id_maps() -> tuple[dict, dict]:
    bs = fpl_api.bootstrap_static()
    return ({e["id"]: e["code"] for e in bs["elements"]},
            {t["id"]: t["code"] for t in bs["teams"]})


@st.cache_data(show_spinner=False)
def img_b64(path_str: str) -> str:
    import base64
    return base64.b64encode(Path(path_str).read_bytes()).decode("ascii")


def shirt_b64(team_id, position) -> str | None:
    """Team kit, like the official FPL squad view -- one of 20 standard kits by
    club, no player mugshots. Returns None for a club whose kit is not in
    assets/ (a newly promoted side), and the chip then renders without an image
    rather than breaking."""
    if team_id is None or pd.isna(team_id):
        return None
    _, team_code = get_id_maps()
    code = team_code.get(int(team_id))
    if code is None:
        return None
    p = fetch_assets.shirt_path(code, is_gk=(position == "GKP"))
    return img_b64(str(p)) if p and p.exists() else None


def money(v) -> str:
    try:
        return f"GBP {float(v):.1f}m"
    except (TypeError, ValueError):
        return "-"


def fig1(draw, height=4.0, width=7.0, projection=None):
    """Draw one charts_core function and hand it to Streamlit. Figures are
    always closed -- an unclosed figure per rerun leaks memory, which matters
    on Streamlit Cloud's free tier."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection=projection) if projection else fig.add_subplot(111)
    try:
        draw(ax)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    finally:
        plt.close(fig)


# ================================================================ sidebar ===

st.sidebar.markdown(
    '<div class="n-dot n-h1" style="font-size:19px">FPL<br>ASSISTANT</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

theme_name = st.sidebar.radio("Theme", list(ui.THEMES.keys()), horizontal=True, key="k_theme")
accent_name = st.sidebar.selectbox("Accent", list(ui.ACCENTS.keys()), index=0, key="k_accent")
st.markdown(ui.css(theme_name, accent_name), unsafe_allow_html=True)
CP = ui.chart_palette(theme_name, accent_name)


def chart(fn, *args, **kwargs):
    """Bind the page palette and a transparent background to a charts_core call."""
    return lambda ax: fn(ax, *args, palette=CP, transparent=True, **kwargs)


st.sidebar.markdown("---")

if st.sidebar.button("Refresh all data", width="stretch"):
    st.cache_data.clear()
    st.rerun()

league_id = st.sidebar.number_input("Mini-league ID", value=DEFAULT_LEAGUE_ID, step=1, key="k_league")

standings = pd.DataFrame()
league_name = "Mini League"
LEAGUE_OK = False
try:
    standings = get_standings(int(league_id))
    league_name = standings.attrs.get("league_name", "Mini League")
    LEAGUE_OK = not standings.empty
except Exception as e:
    st.sidebar.error(f"Couldn't load league {league_id}: {e}")

st.sidebar.caption(f"League: **{league_name}**")

manager_options = {}
if LEAGUE_OK:
    for _, row in standings.sort_values("rank").iterrows():
        manager_options[f"{row['player_name']} ({row['entry_name']})"] = int(row["entry"])

labels = list(manager_options.keys())
default_label = next((k for k, v in manager_options.items() if v == DEFAULT_TEAM_ID), None)
default_idx = labels.index(default_label) if default_label in labels else 0

if labels:
    chosen_label = st.sidebar.selectbox(
        "Viewing as", labels, index=default_idx,
        help="Pick anyone in the mini-league to see the app built around their squad.",
    )
    active_team_id = manager_options[chosen_label]
else:
    chosen_label = "Manual entry"
    active_team_id = st.sidebar.number_input("Team ID (manual)", value=DEFAULT_TEAM_ID, step=1)

_default_event, IS_LIVE_GW = get_active_picks_event()
event = st.sidebar.number_input(
    "Gameweek", value=_default_event, min_value=1, max_value=38, step=1, key="k_gw",
)
# Only trust the live flag when the user has not overridden the auto-detected
# gameweek: picking an old GW manually should never show a "live" badge.
IS_LIVE_GW = IS_LIVE_GW and int(event) == _default_event

st.sidebar.markdown("---")
as_of_label = st.sidebar.radio(
    "Scoring mode",
    ["Frozen (fair)", "Live (in-progress GW)"],
    index=0,
    key="k_asof",
    help=(
        "Frozen scores every player from the last COMPLETED gameweek, so clubs "
        "that have already played the live gameweek get no head start. Live "
        "uses whatever each club has actually played -- useful for one club, "
        "but the numbers are not comparable across clubs."
    ),
)
AS_OF = "frozen" if as_of_label.startswith("Frozen") else "live"

st.sidebar.markdown("---")
st.sidebar.caption(
    "Runs server-side: every visitor shares this host's data pull. "
    "Local Ollama chat only works if Ollama runs on the same machine."
)

# ============================================================ shared data ===

try:
    predictions = get_predictions(AS_OF)
except Exception as e:
    st.error(f"Couldn't score players: {e}")
    st.stop()

PREDICTS_GW = (int(predictions["predicts_gw"].iloc[0])
               if "predicts_gw" in predictions.columns and not predictions.empty else None)

try:
    squad, meta = get_squad(int(active_team_id), int(event))
except Exception as e:
    st.error(f"Couldn't load squad for team {active_team_id}, event {event}: {e}")
    st.stop()

xi, bench = recommend.best_starting_xi(squad)
sellable = squad["sell_price_m"].sum() if "sell_price_m" in squad else meta["value"]
MAE = get_model_mae()

# =================================================================== head ===

st.markdown(f'<div class="n-dot n-h1">{meta["team_name"]}</div>', unsafe_allow_html=True)

bits = [f"Viewing as <b>{chosen_label}</b>"]
if IS_LIVE_GW:
    bits.append(ui.live_badge(f"GW{meta['event']} live"))
if PREDICTS_GW:
    mode_txt = "frozen" if AS_OF == "frozen" else "live, not cross-club comparable"
    bits.append(f"Predictions for <b>GW{PREDICTS_GW}</b> ({mode_txt})")
st.markdown(
    f'<div class="n-h2" style="margin-bottom:16px">{" &nbsp;/&nbsp; ".join(bits)}</div>',
    unsafe_allow_html=True,
)

if AS_OF == "live":
    st.warning(
        "Live mode: clubs that have already played this gameweek are scored on "
        "information the others do not have yet, and their next fixture is the "
        "FOLLOWING gameweek. Rankings across clubs are not like-for-like. "
        "Switch to Frozen for a fair comparison."
    )

live = None
if IS_LIVE_GW:
    try:
        live = get_live_score(squad, int(event), int(active_team_id))
    except Exception as e:
        st.caption(f"Live points unavailable: {e}")

cap_row = squad[squad["is_captain"]] if "is_captain" in squad else pd.DataFrame()
cap_name = cap_row.iloc[0]["web_name"] if not cap_row.empty else "-"

if live:
    st.markdown(ui.tiles([
        ui.tile("Live points", live["live_total"], f"GW{event}", accent=True),
        ui.tile("On bench", live["bench_points"], "not counted"),
        ui.tile("Captain", cap_name, "your pick"),
        ui.tile("Bank", money(meta["bank"])),
        ui.tile("Squad value", money(meta["value"]), f"sellable {money(sellable)}"),
    ]), unsafe_allow_html=True)
else:
    st.markdown(ui.tiles([
        ui.tile("Predicted XI", f"{xi['pred_points_adj'].sum():.1f}", "fixture-adjusted", accent=True),
        ui.tile("Suggested captain", xi.iloc[0]["web_name"], f"{xi.iloc[0]['pred_points_adj']:.1f} pts"),
        ui.tile("Bank", money(meta["bank"])),
        ui.tile("Squad value", money(meta["value"])),
        ui.tile("Sellable", money(sellable), "true sell price"),
    ]), unsafe_allow_html=True)

AI_AVAILABLE = ai_available()

tab_names = ["This Week", "Squad", "Transfers", "Players", "Mini League", "Chips"]
if AI_AVAILABLE:
    tab_names.append("AI")
tab_names.append("Model")
tabs = st.tabs(tab_names)
T = {n: i for i, n in enumerate(tab_names)}

tx = recommend.suggest_transfers(squad, meta["bank"])


# ============================================================== THIS WEEK ===
with tabs[T["This Week"]]:
    cap, vice = xi.iloc[0], xi.iloc[1]

    st.markdown(ui.rule("Decisions"), unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    def _pts_line(row) -> str:
        """'7.8 if he starts · 94% · 7.3 expected' -- pred_points_adj alone
        looks low for premiums because it already factors in the chance they
        don't start; showing both numbers instead of just the shrunk one.
        The two numbers come from separately-trained models (joint vs.
        started-only) that occasionally disagree on nailed players -- when
        that happens "if starts" can come out below "expected", which reads
        as nonsense, so fall back to the single number rather than show it."""
        if_starts = row.get("pred_points_if_starts_adj")
        p = row.get("start_probability")
        adj = row["pred_points_adj"]
        if pd.isna(if_starts) or pd.isna(p) or if_starts < adj:
            return f"{adj:.1f} pts"
        return f"{if_starts:.1f} if he starts &middot; {p * 100:.0f}% &middot; {adj:.1f} expected"

    with d1:
        gap = float(cap["pred_points_adj"]) - float(vice["pred_points_adj"])
        coin = gap < MAE
        st.markdown(ui.card(
            "Captain",
            f"{cap['web_name']}",
            f"{_pts_line(cap)} vs {cap.get('next_fixture', '?')} "
            f"(FDR {cap.get('next_fdr', '?')})<br>"
            + (f"Only {gap:.2f} pts clear of {vice['web_name']} -- inside the model's "
               f"+/-{MAE:.2f} MAE, so this is close to a coin flip."
               if coin else
               f"Clear of {vice['web_name']} by {gap:.2f} pts, beyond the +/-{MAE:.2f} MAE band."),
            hot=True,
        ), unsafe_allow_html=True)
    with d2:
        st.markdown(ui.card(
            "Vice-captain", f"{vice['web_name']}",
            f"{_pts_line(vice)} vs {vice.get('next_fixture', '?')}",
        ), unsafe_allow_html=True)

    d3, d4 = st.columns(2)
    with d3:
        if tx.empty:
            st.markdown(ui.card("Transfer", "Hold",
                                "No upgrade clears the model's threshold within budget."),
                        unsafe_allow_html=True)
        else:
            best = tx.iloc[0]
            flag = best.get("in_price_flag")
            flag = "" if pd.isna(flag) else flag
            st.markdown(ui.card(
                "Best transfer", f"{best['out']} -> {best['in']}",
                f"+{best['pred_gain']:.2f} pts / {money(best['in_cost'])} / "
                f"{money(best['leftover_bank'])} left"
                + (f" {ui.pill(flag, 'good' if flag == 'RISE' else 'bad')}" if flag else ""),
                hot=True,
            ), unsafe_allow_html=True)
    with d4:
        risky = squad[squad["start_probability"] < 0.5] if "start_probability" in squad else pd.DataFrame()
        flagged = (squad[(~squad["status_ok"]) | (squad["status"] == "d")]
                   if "status_ok" in squad else pd.DataFrame())
        names = sorted(set(list(risky.get("web_name", [])) + list(flagged.get("web_name", []))))
        if not names:
            st.markdown(ui.card("Alerts", "All clear",
                                "No availability or rotation flags in your squad."),
                        unsafe_allow_html=True)
        else:
            st.markdown(ui.card("Alerts", f"{len(names)} to check", ", ".join(names)),
                        unsafe_allow_html=True)

    st.markdown(ui.rule("Starting XI"), unsafe_allow_html=True)
    view = st.radio("View", ["Pitch", "Table"], horizontal=True,
                    label_visibility="collapsed", key="k_view")

    def chips_for(df):
        out = []
        for _, p in df.iterrows():
            out.append(ui.player_chip(
                name=p["web_name"],
                meta=f'{p.get("position", "")} / {money(p.get("now_cost_m"))}',
                pts=p.get("pred_points_adj"),
                shirt_b64=shirt_b64(p.get("team_id"), p.get("position")),
                is_captain=bool(p.get("is_captain")),
                is_vice=bool(p.get("is_vice_captain")),
                start_prob=p.get("start_probability"),
            ))
        return out

    if view == "Pitch":
        rows = [chips_for(xi[xi["position"] == pos]) for pos in ("GKP", "DEF", "MID", "FWD")]
        st.markdown(ui.pitch([r for r in rows if r]), unsafe_allow_html=True)
        st.caption("Dot = rotation risk from the minutes classifier (AUC 0.95): "
                   "green nailed, amber probable, red rotation risk.")
        st.markdown(ui.rule("Bench"), unsafe_allow_html=True)
        st.markdown(ui.pitch([chips_for(bench)]), unsafe_allow_html=True)
    else:
        cols = [c for c in ["web_name", "position", "name", "now_cost_m", "pred_points_adj",
                            "start_probability", "next_fixture", "next_fdr"] if c in xi.columns]
        st.dataframe(xi[cols], width="stretch", hide_index=True)
        st.markdown(ui.rule("Bench"), unsafe_allow_html=True)
        st.dataframe(bench[cols], width="stretch", hide_index=True)

    st.markdown(ui.rule("Captain shortlist"), unsafe_allow_html=True)
    fig1(chart(charts_core.captain_uncertainty, xi, MAE), height=3.6)
    in_band = int((xi["pred_points_adj"] >= xi["pred_points_adj"].max() - MAE).sum())
    if in_band > 1:
        st.caption(f"{in_band} candidates sit within one MAE of the leader.")

    if live:
        st.markdown(ui.rule("Live points by player"), unsafe_allow_html=True)
        st.dataframe(live["by_player"], width="stretch", hide_index=True)


# ================================================================== SQUAD ===
with tabs[T["Squad"]]:
    st.markdown(ui.rule("Predicted points"), unsafe_allow_html=True)
    fig1(chart(charts_core.xi_points_bar, xi), height=4.2)

    st.markdown(ui.rule("Value: bought vs now"), unsafe_allow_html=True)
    fig1(chart(charts_core.squad_value_bars, squad), height=4.0)
    vcols = [c for c in ["web_name", "buy_price_m", "now_cost_m", "sell_price_m", "profit_m"]
             if c in squad.columns]
    st.dataframe(squad[vcols].sort_values("profit_m", ascending=False)
                 if "profit_m" in squad else squad[vcols],
                 width="stretch", hide_index=True)

    st.markdown(ui.rule("Rotation risk"), unsafe_allow_html=True)
    if "start_probability" in squad.columns:
        r = squad.copy()

        def band(p):
            if pd.isna(p):
                return "Unknown"
            return "Nailed" if p >= 0.9 else ("Probable" if p >= 0.5 else "Rotation risk")

        r["risk"] = r["start_probability"].apply(band)
        st.dataframe(
            r[["web_name", "position", "start_probability", "pred_points_adj", "risk"]]
            .sort_values("start_probability"),
            width="stretch", hide_index=True,
        )

    st.markdown(ui.rule("Availability"), unsafe_allow_html=True)
    tri = predictions[predictions["id"].isin(set(squad["element"]))]
    tri = tri[(~tri["status_ok"]) | (tri["status"] == "d")]
    if tri.empty:
        st.caption("Nobody in your squad is flagged.")
    else:
        cols = [c for c in ["web_name", "position", "name", "status",
                            "chance_of_playing_next_round", "news", "pred_points_adj"]
                if c in tri.columns]
        st.dataframe(tri[cols], width="stretch", hide_index=True)


# ============================================================== TRANSFERS ===
with tabs[T["Transfers"]]:
    st.markdown(ui.rule("Suggested upgrades"), unsafe_allow_html=True)
    if tx.empty:
        st.info("No clear upgrades within budget.")
    else:
        for _, t in tx.head(5).iterrows():
            flag = t.get("in_price_flag")
            flag = "" if pd.isna(flag) else flag
            st.markdown(ui.card(
                f"+{t['pred_gain']:.2f} pts",
                f"{t['out']} -> {t['in']}",
                f"{t['in_team']} / {money(t['in_cost'])} / sell {money(t['out_sell_price'])} / "
                f"{money(t['leftover_bank'])} left"
                + (f" {ui.pill(flag, 'good' if flag == 'RISE' else 'bad')}" if flag else ""),
            ), unsafe_allow_html=True)
        fig1(chart(charts_core.transfer_gains, tx), height=3.8)
        with st.expander("All suggestions"):
            st.dataframe(tx, width="stretch", hide_index=True)

    st.markdown(ui.rule("Value hunting"), unsafe_allow_html=True)
    fig1(chart(charts_core.value_scatter, predictions), height=4.2)

    st.markdown(ui.rule("Fixture difficulty"), unsafe_allow_html=True)
    fx = get_fixture_matrix(5)
    if fx.empty:
        st.caption("No fixture data available.")
    else:
        fig1(lambda ax: charts_core.fixture_heatmap(ax, fx, palette=CP, transparent=True),
             height=5.0)


# ================================================================ PLAYERS ===
with tabs[T["Players"]]:
    c1, c2 = st.columns([3, 1])
    search = c1.text_input("Search player or club")
    pos_filter = c2.selectbox("Position", ["All", "GKP", "DEF", "MID", "FWD"])
    df = predictions.copy()
    if pos_filter != "All":
        df = df[df["position"] == pos_filter]
    if search:
        s = search.lower()
        df = df[df["web_name"].str.lower().str.contains(s, na=False)
                | df["name"].str.lower().str.contains(s, na=False)]
    cols = [c for c in ["web_name", "position", "name", "now_cost_m", "pred_points_adj",
                        "value_ratio", "start_probability", "form", "selected_by_percent",
                        "next_fixture", "next_fdr", "status", "price_flag"] if c in df.columns]
    st.dataframe(df[cols].head(300), width="stretch", hide_index=True)

    st.markdown(ui.rule("Compare"), unsafe_allow_html=True)
    picks = st.multiselect(
        "Players to compare", predictions["web_name"].tolist(),
        default=predictions["web_name"].head(4).tolist(), max_selections=6,
    )
    if picks:
        fig1(lambda ax: charts_core.radar_comparison(ax, predictions, picks, palette=CP),
             height=4.6, projection="polar")

    st.markdown(ui.rule("Differentials"), unsafe_allow_html=True)
    own_thresh = st.slider("Ownership threshold (%)", 1.0, 20.0, 5.0, 0.5)
    fig1(chart(charts_core.differential_scatter, predictions, own_thresh), height=4.4)

    st.markdown(ui.rule("Price momentum"), unsafe_allow_html=True)
    fig1(chart(charts_core.price_momentum_scatter, predictions), height=4.2)
    my_ids = set(squad["element"])
    pc = [c for c in ["web_name", "position", "name", "now_cost_m",
                      "selected_by_percent", "price_change_percent"]
          if c in predictions.columns]
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Risers**")
        ris = predictions[predictions["price_flag"] == "RISE"].sort_values(
            "price_change_percent", ascending=False).assign(mine="")
        ris.loc[ris["id"].isin(my_ids), "mine"] = "*"
        st.dataframe(ris[pc + ["mine"]].head(15), width="stretch", hide_index=True)
    with r2:
        st.markdown("**Fallers**")
        fal = predictions[predictions["price_flag"] == "FALL"].sort_values(
            "price_change_percent").assign(mine="")
        fal.loc[fal["id"].isin(my_ids), "mine"] = "*"
        st.dataframe(fal[pc + ["mine"]].head(15), width="stretch", hide_index=True)

    with st.expander("Cost x ownership x points (3D)"):
        fig1(lambda ax: charts_core.value_3d(ax, predictions, palette=CP),
             height=4.8, projection="3d")


# ============================================================ MINI LEAGUE ===
with tabs[T["Mini League"]]:
    if not LEAGUE_OK:
        st.info("No league loaded -- enter a valid mini-league ID in the sidebar.")
    else:
        try:
            league_squads = get_league_squads(int(league_id), int(event))
            insights = get_league_insights(int(league_id), int(event), int(active_team_id))
        except Exception as e:
            st.error(f"Couldn't build league squads: {e}")
            league_squads, insights = pd.DataFrame(), {}

        my_row = standings[standings["entry"] == int(active_team_id)]
        my_entry_name = my_row.iloc[0]["entry_name"] if not my_row.empty else ""

        ml = st.tabs(["Table", "Ownership", "Projections", "Form", "Head-to-head"])

        with ml[0]:
            st.dataframe(
                standings[["rank", "entry_name", "player_name", "event_total", "total"]],
                width="stretch", hide_index=True,
            )
            fig1(chart(charts_core.league_standings_bar, standings, my_entry_name), height=4.4)

        with ml[1]:
            if league_squads.empty:
                st.caption("No squad data for this gameweek.")
            else:
                eo = league_extras.effective_ownership(league_squads)
                st.dataframe(
                    eo[["web_name", "position", "owned_pct", "captained_n",
                        "eo_pct", "template_risk"]].head(20),
                    width="stretch", hide_index=True,
                )
                fig1(chart(charts_core.ownership_chart, eo), height=4.0)
                if insights.get("captains") is not None:
                    fig1(chart(charts_core.captaincy_bar, insights["captains"]), height=4.0)
                ci = league_extras.captaincy_impact(league_squads, predictions, int(active_team_id))
                if "error" not in ci:
                    st.markdown(ui.tiles([
                        ui.tile("Your captain", ci["my_captain"]),
                        ui.tile("Edge vs field", f"{ci['edge_vs_league']:+.2f}", "pts", accent=True),
                        ui.tile("League favourite", ci["most_popular"]),
                    ]), unsafe_allow_html=True)

        with ml[2]:
            if league_squads.empty:
                st.caption("No squad data for this gameweek.")
            else:
                proj = league_projection.project_gw_winner(league_squads, standings, int(event))
                st.caption("Based on last known picks -- may be stale.")
                st.dataframe(
                    proj[["projected_rank", "entry_name", "captain_name",
                          "projected_gw_points", "projected_total", "rank_change"]],
                    width="stretch", hide_index=True,
                )
                ltx = league_projection.per_manager_transfer_suggestions(league_squads)
                if not ltx.empty:
                    st.dataframe(ltx.head(20), width="stretch", hide_index=True)

        with ml[3]:
            hist = get_manager_hist(int(league_id))
            n = st.slider("Form window (gameweeks)", 1, 10, 4)
            form = league_extras.league_form(hist, last_n=n)
            fig1(chart(charts_core.league_form_bar, form, n), height=4.0)
            if not hist.empty and "total_points" in hist.columns:
                prog = hist.pivot_table(index="event", columns="entry_name",
                                        values="total_points", aggfunc="last")
                fig1(chart(charts_core.rank_progression, prog, my_entry_name), height=4.4)

        with ml[4]:
            if not labels or league_squads.empty:
                st.caption("Need a loaded league with squads to compare.")
            else:
                h1, h2 = st.columns(2)
                a = h1.selectbox("Manager A", labels, index=default_idx, key="h2h_a")
                b = h2.selectbox("Manager B", labels,
                                 index=1 if len(labels) > 1 else 0, key="h2h_b")
                if a != b:
                    h = league_extras.head_to_head(
                        league_squads, predictions, manager_options[a], manager_options[b])
                    if "error" in h:
                        st.warning(h["error"])
                    else:
                        st.markdown(ui.tiles([
                            ui.tile("Overlap", f"{h['overlap_pct']:.0f}%",
                                    f"{h['shared_n']}/{h['squad_size']} shared"),
                            ui.tile("Projected gap", f"{h['projected_a'] - h['projected_b']:+.2f}",
                                    "A vs B", accent=True),
                        ]), unsafe_allow_html=True)
                        ca, cb = st.columns(2)
                        ca.markdown(f"**Only {h['name_a']}** (C: {h['captain_a']})")
                        ca.dataframe(h["only_a"], width="stretch", hide_index=True)
                        cb.markdown(f"**Only {h['name_b']}** (C: {h['captain_b']})")
                        cb.dataframe(h["only_b"], width="stretch", hide_index=True)
                else:
                    st.info("Pick two different managers.")

                st.markdown(ui.rule("Who owns this player?"), unsafe_allow_html=True)
                q = st.text_input("Player name", key="owner_q")
                if q:
                    st.dataframe(league_extras.player_owners(league_squads, q),
                                 width="stretch", hide_index=True)


# ================================================================== CHIPS ===
with tabs[T["Chips"]]:
    try:
        used = chips.chips_used(int(active_team_id))
    except Exception:
        used = []
    st.markdown(ui.tiles([ui.tile("Chips used", ", ".join(used) if used else "None")]),
                unsafe_allow_html=True)

    bd = chips.detect_blank_double_gameweeks()
    flagged_gw = bd[bd["is_blank"] | bd["is_double"]]
    if flagged_gw.empty:
        st.caption("No blank or double gameweeks in the next 5.")
    else:
        st.dataframe(flagged_gw, width="stretch", hide_index=True)

    sq_p = squad.merge(predictions, left_on="element", right_on="id",
                       how="left", suffixes=("", "_p"))

    st.markdown(ui.rule("Triple Captain"), unsafe_allow_html=True)
    st.dataframe(chips.triple_captain_candidates(sq_p, bd).head(8),
                 width="stretch", hide_index=True)

    st.markdown(ui.rule("Bench Boost"), unsafe_allow_html=True)
    bb = chips.bench_boost_value(sq_p, bench["element"].tolist())
    st.markdown(ui.tiles([
        ui.tile("Bench Boost", f"{bb['bench_boost_value']:.1f}", "extra pts", accent=True),
    ]), unsafe_allow_html=True)
    if bb["risky_bench_players"]:
        st.caption("Risky bench: " + ", ".join(bb["risky_bench_players"]))

    st.markdown(ui.rule("Wildcard / Free Hit"), unsafe_allow_html=True)
    horizon = st.slider("Wildcard horizon (gameweeks)", 1, 10, 5)
    w1, w2 = st.columns(2)
    if w1.button("Compute wildcard", width="stretch"):
        wc = chips.wildcard_squad(int(active_team_id), squad, horizon=horizon)
        st.caption(wc["caveat"])
        st.write(f"{money(wc.get('total_squad_cost', 0))} / {money(wc.get('budget', 0))} "
                 f"({wc['budget_source']}) -- Captain **{wc['captain']}**")
        if not wc["xi"].empty:
            st.dataframe(wc["xi"][["web_name", "position", "name", "now_cost_m", "horizon_score"]],
                         width="stretch", hide_index=True)
    if w2.button("Compute free hit", width="stretch"):
        fh = chips.free_hit_squad(int(active_team_id), squad)
        st.caption(fh["caveat"])
        st.write(f"{money(fh.get('total_squad_cost', 0))} / {money(fh.get('budget', 0))} "
                 f"({fh['budget_source']}) -- Captain **{fh['captain']}**")
        if not fh["xi"].empty:
            st.dataframe(fh["xi"][["web_name", "position", "name", "now_cost_m", "pred_points_adj"]],
                         width="stretch", hide_index=True)


# ===================================================================== AI ===
# Tab only exists when Ollama was reachable at startup, so a cloud deploy with
# no local Ollama simply does not show it rather than showing a dead tab.
if AI_AVAILABLE:
    with tabs[T["AI"]]:
        st.caption(f"Grounded in {chosen_label}'s live squad data via local Ollama (qwen3:8b).")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)
        q = st.chat_input("e.g. Who should I captain this week and why?")
        if q:
            st.session_state.chat_history.append(("user", q))
            with st.chat_message("user"):
                st.write(q)
            ctx = {"meta": meta, "squad": squad, "xi": xi, "bench": bench,
                   "transfers": tx, "standings": standings}
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        ans = llm_assist.ask(q, ctx)
                    except Exception as e:
                        ans = f"Error: {e}"
                st.write(ans)
            st.session_state.chat_history.append(("assistant", ans))


# ================================================================== MODEL ===
with tabs[T["Model"]]:
    st.markdown(ui.rule("How predictions are made"), unsafe_allow_html=True)
    st.markdown(
        "LightGBM regression, one upcoming gameweek. All rolling windows are "
        "leak-free (shifted before that gameweek).\n\n"
        "- **Form:** rolling 3 and 5 GW means of points, minutes, BPS, ICT, xGI, xGC, "
        "goals, assists, clean sheets\n"
        "- **Season-to-date:** expanding mean points, games played\n"
        "- **Last game:** points, minutes\n"
        "- **Fixture:** real venue, both teams' recent scoring/conceding rate\n"
        "- **Player:** cost, position"
    )
    with st.expander("Full feature list"):
        st.code("\n".join(FEATURE_COLS))

    st.markdown(ui.rule("Feature importance"), unsafe_allow_html=True)
    mp = MODEL_DIR / "points_model.txt"
    if mp.exists():
        booster = lgb.Booster(model_file=str(mp))
        imp = pd.Series(booster.feature_importance(importance_type="gain"),
                        index=FEATURE_COLS).sort_values(ascending=False).head(15)
        fig1(chart(charts_core.feature_importance_bar, imp), height=4.4)
        st.caption("Minutes and starts dominate -- rotation risk is the biggest driver of variance.")
    else:
        st.info("models/points_model.txt not found.")

    st.markdown(ui.rule("Validation"), unsafe_allow_html=True)
    st.markdown(
        "1. **Season holdout:** MAE 0.964 vs 1.059 for a naive career-average baseline.\n"
        "2. **Walk-forward:** expanding retrain through a season, 5 GWs ahead each time. "
        "Overall MAE 0.954.\n"
        "3. **Rolling-origin CV:** trained on strictly earlier seasons, validated on each of "
        "the last 4. Mean MAE 0.997, std 0.047. Beats naive every season."
    )
    bp = DATA_DIR / "backtest_results.csv"
    if bp.exists():
        fig1(chart(charts_core.backtest_mae, pd.read_csv(bp)), height=4.0)
    else:
        st.info("data/backtest_results.csv not found.")

    st.markdown(ui.rule("Honest metrics"), unsafe_allow_html=True)
    st.markdown(
        "Raw MAE is dominated by non-playing rows (0 minutes, 0 points). Segmented by "
        "actual minutes: 0 min MAE 0.33, 1-59 min MAE 1.17, 60+ min MAE 2.35. The last "
        "is the real difficulty -- and it means `pred_points_adj` is an expectation that "
        "already discounts for the chance a player does not start, not a prediction of "
        "what he scores if he plays.\n\n"
        "Decision metrics: within-gameweek Spearman 0.72; captain lift over pool average "
        "several points per GW; top-11-by-model beats top-11-by-points-per-game."
    )
    with st.expander("Opponent strength and is_home fix barely moved MAE"):
        st.markdown(
            "MAE moved 0.963 to 0.964 after adding trained opponent-strength features and "
            "fixing a bug where every prediction assumed home advantage. Both were still "
            "correct to make. A per-position check found FPL's own fixture difficulty "
            "rating has near-zero correlation with predicted points."
        )
    with st.expander("Two-stage model tried, didn't beat the single model"):
        st.markdown(
            "Split into a P(minutes>=60) classifier and a points regressor. Classifier AUC "
            "0.950. Regressor beats a matched naive baseline on started rows (2.337 vs "
            "2.502). Combined MAE 1.018, worse than the single model's 0.964. Kept the "
            "single model for points; the classifier is exposed separately as "
            "`start_probability`."
        )
