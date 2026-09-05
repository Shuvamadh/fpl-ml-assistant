"""FPL ML Assistant -- shareable Streamlit version.

Run with:  streamlit run streamlit_app/app.py

Runs entirely server-side (on whoever's machine hosts it), so every visitor
gets the same live data pull and the same local Ollama chat -- friends don't
need Python, PySide6, or Ollama installed themselves, just a browser pointed
at wherever this is running (localhost for same-machine, your LAN IP for
same-network friends, or a tunnel/Streamlit Cloud deploy for anyone else --
see README).

Key feature beyond the desktop app: a "Viewing as" picker in the sidebar
lets anyone in the mini-league select THEIR OWN name and see the whole app
(squad, transfers, chips, AI chat) built around their team instead of the
host's -- one shared app, everyone's own recommendations.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Streamlit's theme (light/dark/auto) is per-viewer and not reliably
# detectable from the server side, so these charts use a transparent
# background + a mid-gray that's legible against either light or dark page
# backgrounds, rather than assuming one theme.
TEXT_COL = "#888888"
GRID_COL = "#888888"


def _style_ax(ax):
    ax.patch.set_alpha(0)
    ax.figure.patch.set_alpha(0)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
        spine.set_alpha(0.4)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.25)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# gui/league_extras.py is deliberately Qt-free (plain DataFrames in/out) so
# it's shared rather than reimplemented here -- but the REST of gui/ imports
# PySide6, which this deployment doesn't install, so only this one file's
# directory goes on the path, not gui/data_bridge.py or anything else in it.
GUI_DIR = Path(__file__).resolve().parent.parent / "gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

# assets/fetch_assets.py is pure requests, no Qt -- reused directly for
# club badges and player photos.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
if str(ASSETS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSETS_DIR))

import lightgbm as lgb

import fetch_assets

import chips
import fpl_api
import league_extras
import league_projection
import llm_assist
import mini_league
import predict
import recommend
import squad_value as squad_value_mod
from features import FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "fpl.ico"
st.set_page_config(
    page_title="FPL ML Assistant",
    page_icon=str(_ICON_PATH) if _ICON_PATH.exists() else "⚽",
    layout="wide",
)

DEFAULT_LEAGUE_ID = 1766517
DEFAULT_TEAM_ID = 8041052


# ---------------------------------------------------------------- caching ---

@st.cache_data(ttl=300, show_spinner=False)
def get_active_picks_event() -> tuple[int, bool]:
    """Which gameweek's picks are the "current" squad, and whether it's
    still being played. bootstrap's is_current event is locked in (deadline
    passed) the moment it starts, so once a gameweek kicks off its picks are
    immediately the real current squad -- not last week's. Probes with the
    default team since deadline timing is identical for everyone.
    Returns (event_id, is_live) where is_live=True means matches are still
    being played (not yet finished) for that event."""
    bs = fpl_api.bootstrap_static(max_age_s=120)
    current_id = fpl_api.current_event(bs)
    current_meta = next((e for e in bs["events"] if e["id"] == current_id), {})
    try:
        fpl_api.entry_picks(DEFAULT_TEAM_ID, current_id)
        return current_id, not current_meta.get("finished", True)
    except Exception:
        prev_id = max(current_id - 1, 1)
        prev_meta = next((e for e in bs["events"] if e["id"] == prev_id), {})
        return prev_id, not prev_meta.get("finished", True)


@st.cache_data(ttl=1800, show_spinner="Scoring ~650 players (live data pull)...")
def get_predictions() -> pd.DataFrame:
    df = predict.score_players()
    # recommend.py/chips.py/league_projection.py all read predictions back
    # from this file rather than taking a DataFrame directly (same contract
    # the desktop GUI's data_bridge.py relies on) -- has to actually be
    # written, not just returned, or every squad/transfer/chip call below
    # fails on a fresh deploy with no pre-existing file on disk.
    predict.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(predict.DATA_DIR / "player_predictions.csv", index=False)
    return df


@st.cache_data(ttl=1800, show_spinner="Loading league standings...")
def get_standings(league_id: int) -> pd.DataFrame:
    return mini_league.league_standings(league_id)


@st.cache_data(ttl=1800, show_spinner="Pulling every manager's squad...")
def get_league_squads(league_id: int, event: int) -> pd.DataFrame:
    standings = get_standings(league_id)
    return mini_league.build_league_squads(standings, event)


@st.cache_data(ttl=1800, show_spinner=False)
def get_league_insights(league_id: int, event: int, my_entry_id: int) -> dict:
    squads = get_league_squads(league_id, event)
    return mini_league.league_insights(squads, my_entry_id)


@st.cache_data(ttl=600, show_spinner="Loading squad...")
def get_squad(team_id: int, event: int):
    squad, meta = recommend.load_squad(team_id, event)
    sv = squad_value_mod.squad_value_report(team_id, squad)
    squad = squad.merge(sv[["element", "buy_price_m", "sell_price_m", "profit_m"]], on="element", how="left")
    return squad, meta


@st.cache_data(ttl=90, show_spinner=False)  # short TTL: this genuinely changes during live matches
def get_live_score(squad: pd.DataFrame, event: int) -> dict:
    return recommend.live_squad_score(squad, event)


@st.cache_data(ttl=1800, show_spinner="Pulling GW-by-GW history for every manager...")
def get_manager_hist(league_id: int) -> pd.DataFrame:
    """One row per (manager, gameweek): points, points_on_bench, value, etc.
    Same shape as gui/data_bridge.py's _manager_gw_history, reimplemented
    here (not imported) since that file pulls in PySide6 at module level and
    this deployment deliberately doesn't install it."""
    import concurrent.futures as cf

    standings = get_standings(league_id)
    rows = []

    def _fetch(row):
        try:
            hist = fpl_api.entry_history(row["entry"])["current"]
            return row["entry_name"], hist
        except Exception:
            return row["entry_name"], []

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for entry_name, hist in ex.map(_fetch, [r for _, r in standings.iterrows()]):
            for h in hist:
                rows.append({**h, "entry_name": entry_name})
    return pd.DataFrame(rows)


def money(v) -> str:
    try:
        return f"£{v:.1f}m"
    except (TypeError, ValueError):
        return "-"


@st.cache_data(ttl=1800, show_spinner=False)
def get_model_mae() -> float:
    """Measured backtest MAE, for honest uncertainty bands on predictions --
    falls back to the documented value if backtest_results.csv is missing."""
    path = Path(__file__).resolve().parent.parent / "data" / "backtest_results.csv"
    try:
        bt = pd.read_csv(path)
        return float(bt["mae_model"].mean())
    except Exception:
        return 0.96


def draw_captain_uncertainty(candidates: pd.DataFrame, mae: float):
    """The GUI_PLAN chart audit's top finding: predictions rendered as bare
    point estimates imply a precision the backtest doesn't support. If the
    #1 vs #2 gap is smaller than the model's own MAE, they're a statistical
    coin flip -- show that honestly instead of a confident ranked list."""
    df = candidates.head(9).sort_values("pred_points_adj")
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(df))))
    y = range(len(df))
    ax.errorbar(
        df["pred_points_adj"], y, xerr=mae, fmt="o", color="#ff4b4b",
        ecolor="#888888", elinewidth=1.5, capsize=3, markersize=6,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["web_name"])
    ax.set_xlabel(f"Predicted points (±{mae:.2f} MAE)")
    leader = df["pred_points_adj"].max()
    ax.axvspan(leader - mae, leader + mae, color="#ff4b4b", alpha=0.08)
    _style_ax(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    in_range = (df["pred_points_adj"] >= leader - mae).sum()
    if in_range > 1:
        st.caption(f"Top {in_range} candidates are within one MAE of each other.")


def rotation_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """M1 from the chart audit: start_probability (the classifier's P(minutes
    >=60), AUC 0.95) is a far better rotation-risk signal than raw past
    minutes, and nothing in the app surfaced it as a risk band before this."""
    out = df.copy()
    def band(p):
        if pd.isna(p):
            return "Unknown"
        if p >= 0.9:
            return "🟢 Nailed"
        if p >= 0.5:
            return "🟡 Probable"
        return "🔴 Rotation risk"
    out["risk"] = out["start_probability"].apply(band)
    return out.sort_values("start_probability", ascending=True)


def availability_triage(predictions: pd.DataFrame, squad_only_ids: set | None = None) -> pd.DataFrame:
    """M2: a direct list of anyone flagged unavailable/doubtful who still
    carries a meaningful predicted-points number -- the model doesn't always
    know what the API's own status flag knows."""
    df = predictions.copy()
    flagged = df[(~df["status_ok"]) | (df["status"] == "d")]
    if squad_only_ids is not None:
        flagged = flagged[flagged["id"].isin(squad_only_ids)]
    cols = ["web_name", "position", "name", "status", "chance_of_playing_next_round", "news", "pred_points_adj"]
    cols = [c for c in cols if c in flagged.columns]
    return flagged[cols].sort_values("pred_points_adj", ascending=False)


@st.cache_resource(show_spinner=False)
def get_id_maps() -> tuple[dict, dict]:
    """element id -> player code, team id -> team code. Needed because
    predictions carry FPL ids but the CDN filenames are keyed by codes."""
    bs = fpl_api.bootstrap_static()
    player_code = {e["id"]: e["code"] for e in bs["elements"]}
    team_code = {t["id"]: t["code"] for t in bs["teams"]}
    return player_code, team_code


def player_image_path(element_id: int, team_id: int, position: str) -> Path | None:
    player_code, team_code = get_id_maps()
    code = player_code.get(int(element_id))
    if code is not None:
        path = fetch_assets.photo_path(code) or fetch_assets.ensure_photo(code)
        if path is not None:
            return path
    tcode = team_code.get(int(team_id))
    if tcode is None:
        return None
    return fetch_assets.shirt_path(tcode, is_gk=(position == "GKP"))


def team_badge_path(team_id: int) -> Path | None:
    _, team_code = get_id_maps()
    tcode = team_code.get(int(team_id))
    return fetch_assets.badge_path(tcode) if tcode is not None else None


def draw_player_grid(df: pd.DataFrame, cols_per_row: int = 5):
    """Card grid: photo/shirt, badge, name, cost, predicted points."""
    rows = [df.iloc[i:i + cols_per_row] for i in range(0, len(df), cols_per_row)]
    for row_df in rows:
        cols = st.columns(cols_per_row)
        for col, (_, player) in zip(cols, row_df.iterrows()):
            with col:
                img_path = player_image_path(player["element" if "element" in player else "id"],
                                              player.get("team_id"), player.get("position"))
                if img_path is not None:
                    st.image(str(img_path), use_container_width=True)
                badge_path = team_badge_path(player.get("team_id"))
                cap = " (C)" if player.get("is_captain") else (" (V)" if player.get("is_vice_captain") else "")
                if badge_path is not None:
                    b1, b2 = st.columns([1, 4])
                    b1.image(str(badge_path), width=24)
                    b2.markdown(f"**{player['web_name']}{cap}**")
                else:
                    st.markdown(f"**{player['web_name']}{cap}**")
                st.caption(f"{player.get('position', '')} | {money(player.get('now_cost_m'))}")
                st.caption(f"{player.get('pred_points_adj', 0):.1f} pts")


# ------------------------------------------------------------------ sidebar ---

st.sidebar.title("⚽ FPL ML Assistant")

if st.sidebar.button("\U0001f504 Refresh all data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

league_id = st.sidebar.number_input("Mini-league ID", value=DEFAULT_LEAGUE_ID, step=1)

try:
    standings = get_standings(int(league_id))
    league_name = standings.attrs.get("league_name", "Mini League")
except Exception as e:
    st.sidebar.error(f"Couldn't load league {league_id}: {e}")
    standings = pd.DataFrame()
    league_name = "Mini League"

st.sidebar.caption(f"League: **{league_name}**")

manager_options = {}
if not standings.empty:
    for _, row in standings.sort_values("rank").iterrows():
        label = f"{row['player_name']} ({row['entry_name']})"
        manager_options[label] = int(row["entry"])

default_label = next((k for k, v in manager_options.items() if v == DEFAULT_TEAM_ID), None)
labels = list(manager_options.keys())
default_idx = labels.index(default_label) if default_label in labels else 0

if labels:
    chosen_label = st.sidebar.selectbox(
        "Viewing as", labels, index=default_idx,
        help="Pick anyone in the mini-league to see the app built around their squad instead.",
    )
    active_team_id = manager_options[chosen_label]
else:
    chosen_label = "Manual entry"
    active_team_id = st.sidebar.number_input("Team ID (manual)", value=DEFAULT_TEAM_ID, step=1)

_default_event, IS_LIVE_GW = get_active_picks_event()
event = st.sidebar.number_input(
    "Gameweek" + (" (live now)" if IS_LIVE_GW else ""),
    value=_default_event, min_value=1, max_value=38, step=1,
)
# only trust the live-in-progress flag when the user hasn't overridden the
# auto-detected gameweek -- picking an old GW manually should never show a
# "live" badge for a match week that's long over.
IS_LIVE_GW = IS_LIVE_GW and int(event) == _default_event

st.sidebar.divider()
st.sidebar.caption(
    "Runs server-side: whoever hosts this page's data pull and local AI chat "
    "are shared by every visitor. Local Ollama chat only works if it's "
    "running on the SAME machine that's serving this app."
)

# ------------------------------------------------------------- shared data ---

try:
    predictions = get_predictions()
except Exception as e:
    st.error(f"Couldn't score players: {e}")
    st.stop()

try:
    squad, meta = get_squad(int(active_team_id), int(event))
except Exception as e:
    st.error(f"Couldn't load squad for team {active_team_id}, event {event}: {e}")
    st.stop()

st.title(f"{meta['team_name']}")
if IS_LIVE_GW:
    st.caption(f"Viewing as **{chosen_label}** | 🔴 GW{meta['event']} live now")
else:
    st.caption(f"Viewing as **{chosen_label}** | Squad entering GW{meta['event'] + 1}")

xi, bench = recommend.best_starting_xi(squad)
sellable = squad["sell_price_m"].sum() if "sell_price_m" in squad else meta["value"]

if IS_LIVE_GW:
    try:
        live = get_live_score(squad, int(event))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bank", money(meta["bank"]))
        c2.metric("Squad Value", money(meta["value"]))
        c3.metric(f"🔴 Live GW{event} Points", live["live_total"])
        c4.metric("Points on bench (live)", live["bench_points"])
        cap_row = squad[squad["is_captain"]]
        cap_name = cap_row.iloc[0]["web_name"] if not cap_row.empty else "-"
        c5.metric("Captain", cap_name)
        with st.expander("Live points by player (this gameweek, updates during matches)"):
            st.dataframe(live["by_player"], use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Couldn't load live points: {e}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bank", money(meta["bank"]))
        c2.metric("Squad Value", money(meta["value"]))
        c3.metric("Sellable Value", money(sellable))
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bank", money(meta["bank"]))
    c2.metric("Squad Value", money(meta["value"]))
    c3.metric("Sellable Value", money(sellable))
    c4.metric("Predicted XI Pts", f"{xi['pred_points_adj'].sum():.1f}")
    c5.metric("Suggested Captain", xi.iloc[0]["web_name"])

AI_AVAILABLE = llm_assist.is_available()

tab_names = [
    "My Squad", "Squad Value", "Transfers", "All Players", "Price Watch",
    "Mini League", "Chips",
]
if AI_AVAILABLE:
    tab_names.append("AI Assistant")
tab_names.append("Model Details")

tabs = st.tabs(tab_names)
tab_idx = {name: i for i, name in enumerate(tab_names)}

# --------------------------------------------------------------- My Squad ---
with tabs[tab_idx["My Squad"]]:
    show_cols = ["web_name", "position", "name", "now_cost_m", "pred_points_adj",
                 "start_probability", "next_fixture", "next_fdr"]
    show_cols = [c for c in show_cols if c in xi.columns]
    view = st.radio("View", ["Grid", "Table"], horizontal=True, label_visibility="collapsed")

    st.subheader("Starting XI")
    if view == "Grid":
        draw_player_grid(xi)
    else:
        st.dataframe(xi[show_cols], use_container_width=True, hide_index=True)

    st.subheader("Bench")
    if view == "Grid":
        draw_player_grid(bench)
    else:
        st.dataframe(bench[show_cols], use_container_width=True, hide_index=True)

    cap, vice = xi.iloc[0], xi.iloc[1]
    st.info(f"**Captain:** {cap['web_name']} vs {cap.get('next_fixture', '?')} "
            f"(FDR {cap.get('next_fdr', '?')}), {cap['pred_points_adj']:.1f} pts\n\n"
            f"**Vice:** {vice['web_name']} vs {vice.get('next_fixture', '?')}, {vice['pred_points_adj']:.1f} pts")

    st.bar_chart(xi.set_index("web_name")["pred_points_adj"], horizontal=True)

    st.divider()
    st.subheader("Captain shortlist")
    mae = get_model_mae()
    draw_captain_uncertainty(xi, mae)

    st.divider()
    st.subheader("Rotation risk")
    if "start_probability" in squad.columns:
        risk = rotation_risk_table(squad)
        st.dataframe(
            risk[["web_name", "position", "start_probability", "pred_points_adj", "risk"]],
            use_container_width=True, hide_index=True,
        )

    triage = availability_triage(predictions, squad_only_ids=set(squad["element"]))
    if not triage.empty:
        st.warning(f"{len(triage)} player(s) flagged unavailable or doubtful")
        st.dataframe(triage, use_container_width=True, hide_index=True)

# ------------------------------------------------------------ Squad Value ---
with tabs[tab_idx["Squad Value"]]:
    st.subheader("Buy price to sell price")
    value_cols = ["web_name", "buy_price_m", "now_cost_m", "sell_price_m", "profit_m"]
    st.dataframe(
        squad[value_cols].sort_values("profit_m", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------- Transfers ---
with tabs[tab_idx["Transfers"]]:
    st.subheader("Suggested upgrades")
    tx = recommend.suggest_transfers(squad, meta["bank"])
    if tx.empty:
        st.success("No clear upgrades found within budget.")
    else:
        st.dataframe(tx, use_container_width=True, hide_index=True)

# ----------------------------------------------------------- All Players ---
with tabs[tab_idx["All Players"]]:
    col1, col2 = st.columns([3, 1])
    search = col1.text_input("Search player or club")
    pos_filter = col2.selectbox("Position", ["All", "GKP", "DEF", "MID", "FWD"])
    df = predictions.copy()
    if pos_filter != "All":
        df = df[df["position"] == pos_filter]
    if search:
        s = search.lower()
        df = df[df["web_name"].str.lower().str.contains(s, na=False) | df["name"].str.lower().str.contains(s, na=False)]
    display_cols = ["web_name", "position", "name", "now_cost_m", "pred_points_adj", "value_ratio",
                     "start_probability", "form", "selected_by_percent", "next_fixture", "next_fdr",
                     "status", "price_flag"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols].head(300), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Differential finder")
    pool = predictions[(predictions["status_ok"]) & (predictions["minutes"] > 0)].copy()
    pool["selected_by_percent"] = pd.to_numeric(pool["selected_by_percent"], errors="coerce")
    pool = pool.dropna(subset=["selected_by_percent", "pred_points_adj"])
    pool = pool[pool["selected_by_percent"] > 0]
    if not pool.empty:
        own_thresh = 5.0
        pts_thresh = pool["pred_points_adj"].median()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(pool["selected_by_percent"], pool["pred_points_adj"], s=14, alpha=0.35, color="#888888")
        diffs = pool[(pool["selected_by_percent"] < own_thresh) & (pool["pred_points_adj"] > pts_thresh)]
        ax.scatter(diffs["selected_by_percent"], diffs["pred_points_adj"], s=40, color="#ff4b4b", zorder=3)
        for _, row in diffs.nlargest(10, "pred_points_adj").iterrows():
            ax.annotate(row["web_name"], (row["selected_by_percent"], row["pred_points_adj"]),
                        color=TEXT_COL, fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.axvline(own_thresh, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(pts_thresh, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Owned % (log scale)")
        ax.set_ylabel("Predicted points")
        ax.set_title(f"Red = differentials (under {own_thresh:.0f}% owned, above-median predicted points)")
        _style_ax(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.expander(f"Differential list ({len(diffs)} players)"):
            st.dataframe(
                diffs[["web_name", "position", "name", "now_cost_m", "selected_by_percent", "pred_points_adj"]]
                .sort_values("pred_points_adj", ascending=False),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.subheader("Availability triage")
    pool_triage = availability_triage(predictions)
    st.dataframe(pool_triage.head(30), use_container_width=True, hide_index=True)

# ----------------------------------------------------------- Price Watch ---
with tabs[tab_idx["Price Watch"]]:
    st.subheader("Price watch")
    price_cols = ["web_name", "position", "name", "now_cost_m", "selected_by_percent", "price_change_percent"]
    my_ids = set(squad["element"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Risers**")
        risers = predictions[predictions["price_flag"] == "RISE"].sort_values(
            "price_change_percent", ascending=False
        )
        risers = risers.assign(mine="")
        risers.loc[risers["id"].isin(my_ids), "mine"] = "⭐ mine"
        st.dataframe(risers[price_cols + ["mine"]].head(20), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**🔴 Fallers**")
        fallers = predictions[predictions["price_flag"] == "FALL"].sort_values(
            "price_change_percent", ascending=True
        )
        fallers = fallers.assign(mine="")
        fallers.loc[fallers["id"].isin(my_ids), "mine"] = "⭐ mine"
        st.dataframe(fallers[price_cols + ["mine"]].head(20), use_container_width=True, hide_index=True)

# ----------------------------------------------------------- Mini League ---
with tabs[tab_idx["Mini League"]]:
    league_squads = get_league_squads(int(league_id), int(event))
    insights = get_league_insights(int(league_id), int(event), int(active_team_id))

    ml_tabs = st.tabs(["Table", "Ownership & Captaincy", "Projections & Transfers", "Fun Stats", "Tools"])

    with ml_tabs[0]:
        st.subheader(f"{league_name} standings")
        if IS_LIVE_GW:
            st.caption(f"GW{event} is live. Points update in near-real-time.")
        st.dataframe(
            standings[["rank", "entry_name", "player_name", "event_total", "total"]],
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(standings.set_index("entry_name")["total"], horizontal=True)

    with ml_tabs[1]:
        eo = league_extras.effective_ownership(league_squads)
        st.subheader("Effective ownership")
        st.dataframe(
            eo[["web_name", "position", "owned_pct", "captained_n", "eo_pct", "template_risk"]].head(20),
            use_container_width=True, hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Captaincy choices**")
            st.dataframe(insights["captains"], use_container_width=True, hide_index=True)
        with col2:
            st.markdown(f"**{chosen_label}'s differentials**")
            st.dataframe(insights["differentials"], use_container_width=True, hide_index=True)

        cap_impact = league_extras.captaincy_impact(league_squads, predictions, int(active_team_id))
        if "error" not in cap_impact:
            st.markdown(f"**Captaincy edge vs the field**")
            edge = cap_impact["edge_vs_league"]
            st.metric(
                f"{chosen_label}'s captain: {cap_impact['my_captain']}",
                f"{edge:+.2f} pts vs league-average captain",
                delta=f"{edge:+.2f}",
            )
            st.caption(f"Most popular captain in the league: {cap_impact['most_popular']}")
            with st.expander("Captaincy split across the league"):
                st.dataframe(cap_impact["split"], use_container_width=True, hide_index=True)

    with ml_tabs[2]:
        st.subheader("Projected GW winner")
        st.caption("Based on last known picks, may be stale.")
        proj = league_projection.project_gw_winner(league_squads, standings, int(event))
        st.dataframe(
            proj[["projected_rank", "entry_name", "captain_name", "projected_gw_points", "projected_total", "rank_change"]],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Per-manager transfer suggestions")
        league_tx = league_projection.per_manager_transfer_suggestions(league_squads)
        st.dataframe(league_tx.head(20) if not league_tx.empty else league_tx, use_container_width=True, hide_index=True)

    with ml_tabs[3]:
        banter = league_projection.banter_stats(league_squads, standings, insights)
        st.subheader("Banter stats")
        st.dataframe(banter["per_manager_stats"], use_container_width=True, hide_index=True)
        st.caption(banter["known_gap"])
        st.markdown("**Template adherence**")
        st.dataframe(banter["template_adherence"], use_container_width=True, hide_index=True)

        st.divider()
        manager_hist = get_manager_hist(int(league_id))
        form_n = st.slider("Recent form window (gameweeks)", 1, 10, 4)
        form = league_extras.league_form(manager_hist, last_n=form_n)
        st.markdown(f"**League form, last {form_n} gameweeks**")
        st.dataframe(form, use_container_width=True, hide_index=True)
        if not form.empty:
            st.bar_chart(form.set_index("entry_name")["recent_points"], horizontal=True)

    with ml_tabs[4]:
        st.subheader("Head-to-head")
        h2h_col1, h2h_col2 = st.columns(2)
        h2h_a = h2h_col1.selectbox("Manager A", labels, index=default_idx, key="h2h_a")
        h2h_b = h2h_col2.selectbox(
            "Manager B", labels, index=min(1, len(labels) - 1) if len(labels) > 1 else 0, key="h2h_b",
        )
        if h2h_a and h2h_b and h2h_a != h2h_b:
            h2h = league_extras.head_to_head(
                league_squads, predictions, manager_options[h2h_a], manager_options[h2h_b],
            )
            if "error" in h2h:
                st.warning(h2h["error"])
            else:
                st.caption(f"{h2h['shared_n']} of {h2h['squad_size']} players shared ({h2h['overlap_pct']:.0f}% overlap)")
                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"**Only {h2h['name_a']}** (captain: {h2h['captain_a']}), {h2h['unique_pred_a']} pred pts")
                    st.dataframe(h2h["only_a"], use_container_width=True, hide_index=True)
                with colB:
                    st.markdown(f"**Only {h2h['name_b']}** (captain: {h2h['captain_b']}), {h2h['unique_pred_b']} pred pts")
                    st.dataframe(h2h["only_b"], use_container_width=True, hide_index=True)
                st.metric("Projected gap (A vs B, starters)", f"{h2h['projected_a'] - h2h['projected_b']:+.2f} pts")
        elif h2h_a == h2h_b:
            st.info("Pick two different managers to compare.")

        st.divider()
        st.subheader("Who owns this player?")
        query = st.text_input("Player name")
        if query:
            owners = league_extras.player_owners(league_squads, query)
            st.dataframe(owners, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- Chips ---
with tabs[tab_idx["Chips"]]:
    used = chips.chips_used(int(active_team_id))
    st.write(f"Chips used this season: {', '.join(used) if used else 'none'}")

    bd = chips.detect_blank_double_gameweeks()
    flagged = bd[bd["is_blank"] | bd["is_double"]]
    if flagged.empty:
        st.info("No blank or double gameweeks detected in the next 5 GWs.")
    else:
        st.warning("Blank/double gameweeks ahead:")
        st.dataframe(flagged, use_container_width=True, hide_index=True)

    horizon = st.slider("Wildcard horizon (gameweeks)", 1, 10, 5)
    if st.button("Compute wildcard squad"):
        wc = chips.wildcard_squad(int(active_team_id), squad, horizon=horizon)
        st.caption(wc["caveat"])
        st.write(f"Budget: {money(wc.get('total_squad_cost', 0))} / {money(wc.get('budget', 0))} "
                 f"({wc['budget_source']})")
        if not wc["xi"].empty:
            st.dataframe(
                wc["xi"][["web_name", "position", "name", "now_cost_m", "horizon_score"]],
                use_container_width=True, hide_index=True,
            )
            st.write(f"Captain: **{wc['captain']}**")

    if st.button("Compute free hit squad"):
        fh = chips.free_hit_squad(int(active_team_id), squad)
        st.caption(fh["caveat"])
        st.write(f"Budget: {money(fh.get('total_squad_cost', 0))} / {money(fh.get('budget', 0))} "
                 f"({fh['budget_source']})")
        if not fh["xi"].empty:
            st.dataframe(
                fh["xi"][["web_name", "position", "name", "now_cost_m", "pred_points_adj"]],
                use_container_width=True, hide_index=True,
            )
            st.write(f"Captain: **{fh['captain']}**")

    st.subheader("Triple Captain candidates")
    squad_preds = squad.merge(predictions, left_on="element", right_on="id", how="left", suffixes=("", "_p"))
    tc = chips.triple_captain_candidates(squad_preds, bd)
    st.dataframe(tc.head(8), use_container_width=True, hide_index=True)

    st.subheader("Bench Boost value")
    bb = chips.bench_boost_value(squad_preds, bench["element"].tolist())
    st.write(f"Bench Boost would add **{bb['bench_boost_value']:.1f} pts**")
    if bb["risky_bench_players"]:
        st.warning(f"Risky bench players (low chance of playing): {', '.join(bb['risky_bench_players'])}")

# ---------------------------------------------------------- AI Assistant ---
# Tab only exists at all when Ollama was reachable at startup (see tab_names
# above) -- on a cloud deploy with no local Ollama, this whole section is
# skipped and the tab simply isn't there, rather than showing a dead tab.
if AI_AVAILABLE:
    with tabs[tab_idx["AI Assistant"]]:
        st.caption(f"Grounded in {chosen_label}'s live squad/predictions data, via local Ollama (qwen3:8b).")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        question = st.chat_input("e.g. Who should I captain this week and why?")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)
            data_ctx = {
                "meta": meta, "squad": squad, "xi": xi, "bench": bench,
                "transfers": recommend.suggest_transfers(squad, meta["bank"]),
                "standings": standings,
                "insights": get_league_insights(int(league_id), int(event), int(active_team_id)),
            }
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = llm_assist.ask(question, data_ctx)
                    except Exception as e:
                        answer = f"Error: {e}"
                st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))

# ------------------------------------------------------------ Model Details ---
with tabs[tab_idx["Model Details"]]:
    st.header("How the predictions are made")
    st.markdown("**Model:** LightGBM, regression, predicts points for one upcoming gameweek.")

    st.subheader("Features")
    st.markdown(
        "All rolling/expanding windows are leak-free (shifted before that gameweek).\n\n"
        "- **Form:** rolling 3 and 5 gameweek averages of points, minutes, BPS, ICT, "
        "expected goal involvement, expected goals conceded, goals, assists, clean sheets\n"
        "- **Season-to-date:** expanding mean points and games played\n"
        "- **Last game:** points and minutes\n"
        "- **Fixture:** real venue, both teams' recent scoring/conceding rate\n"
        "- **Player:** cost, position"
    )
    with st.expander("Full feature list"):
        st.code("\n".join(FEATURE_COLS))

    st.subheader("Feature importance")
    model_path = MODEL_DIR / "points_model.txt"
    if model_path.exists():
        booster = lgb.Booster(model_file=str(model_path))
        imp = pd.Series(booster.feature_importance(importance_type="gain"), index=FEATURE_COLS)
        imp = imp.sort_values(ascending=False).head(15)
        st.bar_chart(imp, horizontal=True)
        st.caption("Minutes and starts dominate. Rotation risk is the biggest driver of point variance.")
    else:
        st.info("models/points_model.txt not found.")

    st.subheader("Validation")
    st.markdown(
        "1. **Season holdout:** MAE 0.964 vs 1.059 for a naive career-average baseline.\n"
        "2. **Walk-forward:** expanding retrain through a season, predicting 5 GWs "
        "ahead each time. Overall MAE 0.954. Accuracy improves as the season progresses.\n"
        "3. **Rolling-origin CV:** trained on strictly earlier seasons, validated on "
        "each of the last 4. Mean MAE 0.997, std 0.047. Beats naive in every season."
    )

    backtest_path = DATA_DIR / "backtest_results.csv"
    if backtest_path.exists():
        bt = pd.read_csv(backtest_path)
        st.markdown("**Walk-forward MAE by gameweek**")
        st.line_chart(bt.set_index("GW")[["mae_model", "mae_naive"]])
    else:
        st.info("data/backtest_results.csv not found.")

    rolling_path = DATA_DIR / "rolling_origin_results.csv"
    if rolling_path.exists():
        ro = pd.read_csv(rolling_path)
        st.markdown("**Rolling-origin CV by season**")
        st.bar_chart(ro.set_index("val_season")[["mae_model", "mae_naive"]])
    else:
        st.info("data/rolling_origin_results.csv not found.")

    st.subheader("Honest metrics")
    st.markdown(
        "Raw MAE is dominated by non-playing rows (0 minutes, 0 points). Segmented "
        "by actual minutes: 0 min MAE 0.33, 1 to 59 min MAE 1.17, 60+ min MAE 2.35. "
        "The last one is the real difficulty.\n\n"
        "Decision metrics:\n"
        "- Within-gameweek Spearman rank correlation: 0.72\n"
        "- Captain lift over pool average: several points per gameweek\n"
        "- Top-11-by-model vs top-11-by-points-per-game: model wins"
    )

    st.subheader("Two findings reported as measured")
    with st.expander("Opponent strength and is_home fix barely moved MAE"):
        st.markdown(
            "MAE moved from 0.963 to 0.964 after adding trained opponent-strength "
            "features and fixing a bug where every prediction assumed home advantage. "
            "Both fixes were still correct to make. A per-position check found FPL's "
            "own fixture difficulty rating has close to zero correlation with "
            "predicted points, even controlling for position."
        )
    with st.expander("Two-stage model tried, didn't beat the single model"):
        st.markdown(
            "Split into a P(minutes >= 60) classifier and a points regressor. "
            "Classifier AUC 0.950. Regressor beats a matched naive baseline on "
            "started rows (2.337 vs 2.502). Combined MAE was 1.018, worse than the "
            "single model's 0.964. Kept the single model for points. The classifier "
            "is exposed separately as start_probability."
        )

    st.caption("Full writeup in README.md.")
