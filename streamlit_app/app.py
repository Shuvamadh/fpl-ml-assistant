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

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lightgbm as lgb

import chips
import fpl_api
import league_projection
import llm_assist
import mini_league
import predict
import recommend
import squad_value as squad_value_mod
from features import FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="FPL ML Assistant", page_icon="⚽", layout="wide")

DEFAULT_LEAGUE_ID = 1766517
DEFAULT_TEAM_ID = 8041052


# ---------------------------------------------------------------- caching ---

@st.cache_data(ttl=1800, show_spinner=False)
def get_current_event() -> int:
    bs = fpl_api.bootstrap_static()
    ev = fpl_api.current_event(bs)
    return max(ev - 1, 1)  # "last finished" event, same convention as the GUI


@st.cache_data(ttl=1800, show_spinner="Scoring ~650 players (live data pull)...")
def get_predictions() -> pd.DataFrame:
    return predict.score_players()


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


def money(v) -> str:
    try:
        return f"£{v:.1f}m"
    except (TypeError, ValueError):
        return "-"


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
        label = f"{row['player_name']} — {row['entry_name']}"
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

event = st.sidebar.number_input("Last finished gameweek", value=get_current_event(), min_value=1, max_value=38, step=1)

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
st.caption(f"Viewing as **{chosen_label}** | Squad entering GW{meta['event'] + 1}")

xi, bench = recommend.best_starting_xi(squad)
sellable = squad["sell_price_m"].sum() if "sell_price_m" in squad else meta["value"]

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
    st.subheader("Starting XI")
    st.dataframe(xi[show_cols], use_container_width=True, hide_index=True)
    st.subheader("Bench")
    st.dataframe(bench[show_cols], use_container_width=True, hide_index=True)

    cap, vice = xi.iloc[0], xi.iloc[1]
    st.info(f"**Captain:** {cap['web_name']} vs {cap.get('next_fixture', '?')} "
            f"(FDR {cap.get('next_fdr', '?')}) — {cap['pred_points_adj']:.1f} pts\n\n"
            f"**Vice:** {vice['web_name']} vs {vice.get('next_fixture', '?')} — {vice['pred_points_adj']:.1f} pts")

    st.bar_chart(xi.set_index("web_name")["pred_points_adj"], horizontal=True)

# ------------------------------------------------------------ Squad Value ---
with tabs[tab_idx["Squad Value"]]:
    st.subheader("Buy price -> real sell price (FPL's half-profit-on-rise rule applied)")
    value_cols = ["web_name", "buy_price_m", "now_cost_m", "sell_price_m", "profit_m"]
    st.dataframe(
        squad[value_cols].sort_values("profit_m", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------- Transfers ---
with tabs[tab_idx["Transfers"]]:
    st.subheader("Suggested upgrades within real sellable budget")
    tx = recommend.suggest_transfers(squad, meta["bank"])
    if tx.empty:
        st.success("No clear upgrades found within budget — squad looks efficient per the model.")
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

# ----------------------------------------------------------- Price Watch ---
with tabs[tab_idx["Price Watch"]]:
    st.subheader("FPL's own transfer-momentum signal")
    watch = predictions[predictions["price_flag"] != ""].sort_values(
        "price_signal", key=lambda s: s.abs(), ascending=False
    )
    price_cols = ["web_name", "position", "name", "now_cost_m", "selected_by_percent", "price_flag", "price_change_percent"]
    st.dataframe(watch[price_cols], use_container_width=True, hide_index=True)

# ----------------------------------------------------------- Mini League ---
with tabs[tab_idx["Mini League"]]:
    st.subheader(f"{league_name} standings")
    st.dataframe(
        standings[["rank", "entry_name", "player_name", "event_total", "total"]],
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(standings.set_index("entry_name")["total"], horizontal=True)

    league_squads = get_league_squads(int(league_id), int(event))
    insights = get_league_insights(int(league_id), int(event), int(active_team_id))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Most owned (template)**")
        st.dataframe(insights["ownership"].head(10), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Captaincy choices**")
        st.dataframe(insights["captains"], use_container_width=True, hide_index=True)
    with col3:
        st.markdown(f"**{chosen_label}'s differentials**")
        st.dataframe(insights["differentials"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Projected GW winner (⚠️ stale — based on last known picks)")
    proj = league_projection.project_gw_winner(league_squads, standings, int(event))
    st.dataframe(
        proj[["projected_rank", "entry_name", "captain_name", "projected_gw_points", "projected_total", "rank_change"]],
        use_container_width=True, hide_index=True,
    )

    st.subheader("Per-manager transfer suggestions (assumed bank = £0.0m)")
    league_tx = league_projection.per_manager_transfer_suggestions(league_squads)
    st.dataframe(league_tx.head(20) if not league_tx.empty else league_tx, use_container_width=True, hide_index=True)

    st.subheader("Banter stats")
    banter = league_projection.banter_stats(league_squads, standings, insights)
    st.dataframe(banter["per_manager_stats"], use_container_width=True, hide_index=True)
    st.caption(banter["known_gap"])
    st.markdown("**Template adherence**")
    st.dataframe(banter["template_adherence"], use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- Chips ---
with tabs[tab_idx["Chips"]]:
    used = chips.chips_used(int(active_team_id))
    st.write(f"Chips used this season: {', '.join(used) if used else 'none'}")

    bd = chips.detect_blank_double_gameweeks()
    flagged = bd[bd["is_blank"] | bd["is_double"]]
    if flagged.empty:
        st.info("No blank/double gameweeks detected in the next 5 GWs — normal fixture schedule ahead.")
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

    st.markdown(
        "**Model**: LightGBM (gradient-boosted decision trees), regression objective, "
        "predicting a player's FPL points in a single upcoming gameweek."
    )

    st.subheader("Features (what the model actually sees before a gameweek is played)")
    st.markdown(
        "Every feature below is deliberately **leak-free**: rolling/expanding windows "
        "are shifted so a row only ever uses information available *strictly before* "
        "that gameweek kicked off.\n\n"
        "- **Form**: rolling 3- and 5-gameweek averages of points, minutes, BPS, ICT "
        "index, expected goal involvement, expected goals conceded, goals, assists, "
        "clean sheets\n"
        "- **Season-to-date**: expanding mean points and games played so far this season\n"
        "- **Most recent game**: last gameweek's points and minutes\n"
        "- **Fixture context**: real next-fixture home/away venue, and *both* teams' "
        "recent scoring/conceding rate — own team **and** opponent, recovered from "
        "the fixture list itself rather than an external team-strength table (robust to "
        "promoted/relegated clubs shifting team IDs between seasons)\n"
        "- **Player**: cost, position"
    )
    with st.expander("See the exact feature list"):
        st.code("\n".join(FEATURE_COLS))

    st.subheader("Feature importance (from the trained model)")
    model_path = MODEL_DIR / "points_model.txt"
    if model_path.exists():
        booster = lgb.Booster(model_file=str(model_path))
        imp = pd.Series(booster.feature_importance(importance_type="gain"), index=FEATURE_COLS)
        imp = imp.sort_values(ascending=False).head(15)
        st.bar_chart(imp, horizontal=True)
        st.caption(
            "Minutes/starts dominate — as expected, rotation risk is the single "
            "biggest driver of FPL point variance, more than underlying quality stats. "
            "That's also why there's a separate start_probability column elsewhere in "
            "this app: a dedicated classifier for \"will they play\" turned out to be "
            "excellent (AUC 0.95) even though naively combining it with the points "
            "regressor did not beat this single model (see below)."
        )
    else:
        st.info("models/points_model.txt not found in this deployment.")

    st.subheader("Validated three ways, not one")
    st.markdown(
        "1. **Season holdout**: train on all other seasons, predict an entirely unseen "
        "one. MAE **0.964** vs **1.059** for a naive \"career average\" baseline.\n\n"
        "2. **Walk-forward validation**: expanding-window retrain through a season, "
        "predicting 5-gameweek blocks ahead each time, never looking forward — "
        "simulates exactly how the model gets used in production. Overall MAE **0.954**, "
        "and — the important part — accuracy *improves* as the season "
        "progresses, the correct signature of a model actually using accumulating "
        "in-season signal rather than overfitting to history.\n\n"
        "3. **Rolling-origin cross-season CV**: train on strictly-earlier seasons only, "
        "validate on each of the last 4 seasons in turn, to check whether the "
        "season-holdout number is stable or a lucky draw. Result: mean MAE 0.997, std "
        "0.047 — stable, and the model beats the naive baseline in every single "
        "season tested, not just on average."
    )

    backtest_path = DATA_DIR / "backtest_results.csv"
    if backtest_path.exists():
        bt = pd.read_csv(backtest_path)
        st.markdown("**Walk-forward MAE by gameweek** (lower is better; model vs naive baseline)")
        st.line_chart(bt.set_index("GW")[["mae_model", "mae_naive"]])
    else:
        st.info("data/backtest_results.csv not found in this deployment.")

    rolling_path = DATA_DIR / "rolling_origin_results.csv"
    if rolling_path.exists():
        ro = pd.read_csv(rolling_path)
        st.markdown("**Rolling-origin cross-season CV** (each bar = train on strictly-earlier seasons, validate on that season)")
        st.bar_chart(ro.set_index("val_season")[["mae_model", "mae_naive"]])
    else:
        st.info("data/rolling_origin_results.csv not found in this deployment.")

    st.subheader("The honest metrics, not just MAE")
    st.markdown(
        "Raw MAE is dominated by the ~55-60% of rows that are non-playing (0 minutes, "
        "0 points) and trivially easy to get right. Segmented by actual minutes played "
        "on the held-out season: **0-min MAE ≈ 0.33**, **1-59min MAE ≈ 1.17**, "
        "**60+min (started) MAE ≈ 2.35** — the last one is the real difficulty, "
        "and the number that actually matters for lineup decisions.\n\n"
        "Decision-quality metrics (what the app actually does with these predictions "
        "— ranking and picking, not just minimizing an average error):\n"
        "- **Within-gameweek Spearman rank correlation ≈ 0.72** — how well the "
        "model *orders* players, which is what picking and captaining actually needs, "
        "more than absolute error\n"
        "- **Captain lift over the pool average**: the model's top captain pick "
        "consistently outscores an average player in the pool by several points per "
        "gameweek\n"
        "- **Unconstrained top-11-by-model vs top-11-by-points-per-game**: the model's "
        "ranking beats the simplest baseline a manager could use with no tool at all"
    )

    st.subheader("Two honest negative-ish findings, reported rather than hidden")
    with st.expander("1. Opponent-strength features and the is_home fix barely moved the headline MAE"):
        st.markdown(
            "Aggregate MAE barely moved (0.963 → 0.964, statistically "
            "indistinguishable) after adding trained opponent-strength features and "
            "fixing a real bug where every prediction was computed as if the player "
            "were always at home. Both fixes were still the right calls — a known "
            "systematic bug is worth fixing regardless of whether it moves the headline "
            "number, and a principled trained signal beats an arbitrary hand-tuned one "
            "even at parity — but a controlled per-position check confirmed FPL's "
            "own fixture-difficulty rating has close to **zero** measured correlation "
            "with predicted points, even controlling for position. This matches a known "
            "critique in the wider FPL analytics community: FDR is a coarse "
            "team-strength proxy that ignores injuries, tactical matchups, and "
            "individual role, and its marginal predictive power for a single gameweek "
            "is genuinely weak once a player's own form/quality is accounted for."
        )
    with st.expander("2. The two-stage (hurdle) model was tried and didn't beat the single model"):
        st.markdown(
            "Split the prediction into a P(minutes≥60) classifier and an "
            "E[points | started] regressor, motivated by feature importance suggesting "
            "the single model is mostly a will-he-play classifier wearing a regressor's "
            "clothes. The classifier alone is genuinely excellent (**AUC 0.950**), and "
            "the regressor beats a same-rows-only naive baseline on started rows (MAE "
            "2.337 vs 2.502). But the naive combination — P(started) × "
            "E[points|started] + (1-P(started)) × flat_baseline — has "
            "**worse** overall MAE (1.018) than the single joint model (0.964): a flat "
            "constant for the \"didn't start\" case is cruder than what the single "
            "model already learns implicitly by conditioning continuously on form. The "
            "single-stage model stayed the production predictor; the classifier's "
            "P(started) is exposed separately as the start_probability column since "
            "it's a genuinely useful rotation-risk signal on its own, just not a better "
            "points predictor when naively combined."
        )

    st.caption(
        "Full technical writeup, gotchas, and file-by-file pipeline docs in this "
        "project's README.md."
    )
