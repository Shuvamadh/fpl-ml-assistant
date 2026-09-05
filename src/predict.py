"""Score every current FPL player with the trained model, using this season's
gameweeks-so-far (pulled live from the official API's per-player endpoint,
since the community archive lags by a gameweek early in the season)."""
import concurrent.futures as cf
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURE_COLS, NUMERIC_ROLL_COLS, ROLL_WINDOWS, _team_game_table, build_team_id_map
from fixtures_fdr import next_fixture_summary, target_event, upcoming_fixtures_by_team
from fpl_api import bootstrap_static, player_summary

# small residual post-hoc adjustment on top of the model's own learned
# opponent-strength features (see build_current_features -- opp_team_gf/ga
# are now trained-in, not just this multiplier). Kept small and centred near
# 1.0 since the model already captures most of the fixture-difficulty signal;
# this just nudges for anything the rolling-goals proxy misses.
FDR_ADJUSTMENT = {1: 1.08, 2: 1.04, 3: 1.00, 4: 0.96, 5: 0.92}

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_current_gws(player_ids: list[int], id_to_team: dict[int, str],
                       id_to_team_id: dict[int, int] | None = None, workers: int = 16) -> pd.DataFrame:
    """Pull element-summary history for every player and stack into a
    merged_gw-style dataframe: one row per player per gameweek played."""
    rows = []
    id_to_team_id = id_to_team_id or {}

    def _fetch(pid):
        try:
            return pid, player_summary(pid)["history"]
        except Exception:
            return pid, []

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, history in ex.map(_fetch, player_ids):
            for h in history:
                h = dict(h)
                h["team"] = id_to_team.get(pid)
                h["team_id"] = id_to_team_id.get(pid)
                rows.append(h)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"round": "GW"})
    df["season"] = "live"
    df["team_h_score"] = pd.to_numeric(df["team_h_score"], errors="coerce")
    df["team_a_score"] = pd.to_numeric(df["team_a_score"], errors="coerce")
    df["was_home"] = df["was_home"].astype(bool)
    return df


def build_current_features(gws: pd.DataFrame, for_event: int | None = None) -> pd.DataFrame:
    """Same rolling logic as features.py, but keep only the LATEST row per
    player (i.e. features describing 'form entering the next gameweek'),
    with NO shift since we want form INCLUDING the most recent GW played.

    for_event is the gameweek being predicted. Rows from that gameweek onward
    are dropped before any rolling window is computed, so every player is
    described by the same number of completed gameweeks. Mid-gameweek this
    matters a lot: without it, a club that has already played the live
    gameweek carries that result in prev_gw_points and in every _r3/_r5 mean,
    so its players get scored on information nobody else's players have yet.
    """
    df = gws.sort_values(["element", "GW"]).reset_index(drop=True)
    if for_event is not None:
        df = df[df["GW"] < for_event].reset_index(drop=True)
    for col in NUMERIC_ROLL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    grp = df.groupby("element")
    for w in ROLL_WINDOWS:
        # min_periods=1 meant a "5-game rolling mean" was often a single
        # match, so one good game read as settled form -- worst in GW1-4 and
        # for anyone just back from injury. Requiring 2 observations leaves
        # NaN instead, which the model handles natively and which routes
        # genuinely history-less players to the position-median fallback in
        # score_players() rather than to a confident number built on one game.
        min_obs = 1 if w <= 2 else 2
        for col in NUMERIC_ROLL_COLS:
            df[f"{col}_r{w}"] = grp[col].transform(
                lambda s, w=w, m=min_obs: s.rolling(w, min_periods=m).mean()
            )
    df["season_pts_mean_prior"] = grp["total_points"].transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    df["season_gp_prior"] = grp["total_points"].transform(
        lambda s: s.expanding(min_periods=1).count()
    )
    df["prev_gw_points"] = df["total_points"]
    df["prev_gw_minutes"] = df["minutes"]

    tg = _team_game_table(df)
    opp_cols = [c for c in tg.columns if c.startswith("team_g")]
    tg2 = tg.copy()
    for w in ROLL_WINDOWS:
        tg2[f"team_gf_{w}"] = (
            tg2.groupby("team")["goals_for"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        tg2[f"team_ga_{w}"] = (
            tg2.groupby("team")["goals_against"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
    latest_team_form = tg2.sort_values("GW").groupby("team").tail(1)
    df = df.merge(
        latest_team_form[["team"] + opp_cols].rename(columns={c: f"self_{c}" for c in opp_cols}),
        on="team",
        how="left",
    )

    # opponent's recent scoring/defensive form for the NEXT fixture (not a
    # played one) -- look up the next opponent's team id via the fixtures
    # endpoint, then pull that team's own latest rolling form.
    id_map = build_team_id_map(df)
    latest_with_id = latest_team_form.merge(id_map, on=["season", "team"], how="left")
    next_fx = upcoming_fixtures_by_team(n=1, from_event=for_event)[["team", "opponent"]].rename(
        columns={"team": "team_id", "opponent": "next_opponent_id"}
    )
    df = df.merge(next_fx, on="team_id", how="left")
    opp_form = latest_with_id[["team_id"] + opp_cols].rename(
        columns={"team_id": "next_opponent_id", **{c: f"opp_{c}" for c in opp_cols}}
    )
    df = df.merge(opp_form, on="next_opponent_id", how="left")

    # real next-fixture venue, not a neutral placeholder -- the model was
    # trained on true is_home, so a wrong/constant value here biased every
    # single prediction (previously always "1", i.e. every player looked
    # like they were at home next gameweek).
    fdr = next_fixture_summary(n=1, from_event=for_event)[["team", "is_home"]].rename(
        columns={"team": "team_id"}
    )
    df = df.merge(fdr, on="team_id", how="left")
    df["is_home"] = df["is_home"].fillna(True).astype(int)
    df["cost"] = df["value"] / 10.0

    latest = df.sort_values("GW").groupby("element").tail(1).copy()
    return latest


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_DIR / "points_model.txt"))


def load_minutes_classifier() -> lgb.Booster | None:
    """P(minutes>=60) classifier from train_model.train_two_stage() -- AUC
    ~0.95, a genuinely strong rotation-risk signal on its own. Exposed as a
    separate `start_probability` column, NOT blended into pred_points: a
    from-scratch comparison found the naive two-stage combination (p*E[pts|
    started] + (1-p)*flat_baseline) had WORSE overall MAE (1.02) than the
    single joint model (0.96) -- a flat baseline for the "didn't start" case
    is cruder than what the joint model already learns implicitly. Optional:
    older model directories won't have this file yet.
    """
    path = MODEL_DIR / "minutes_classifier.txt"
    return lgb.Booster(model_file=str(path)) if path.exists() else None


def load_started_regressor() -> lgb.Booster | None:
    """E[points | started] from train_model.train_two_stage() -- trained only
    on minutes>=60 rows. `pred_points` (the joint model) already factors in
    the chance a player doesn't start, which shrinks premiums relative to
    what they'd score if guaranteed 90 minutes. This gives the "if he starts"
    number separately so the UI can show both instead of only the shrunk one.
    Optional: older model directories won't have this file yet.
    """
    path = MODEL_DIR / "points_regressor_started.txt"
    return lgb.Booster(model_file=str(path)) if path.exists() else None


def score_players(as_of: str = "frozen") -> pd.DataFrame:
    """Score every player for one upcoming gameweek.

    as_of="frozen" (default) scores everyone from the last fully completed
    gameweek, so no club has a head start on any other. This is the only mode
    that produces a comparable ranking while a gameweek is in progress.

    as_of="live" uses whatever each player's club has actually played,
    including a gameweek still under way. Useful for looking at one club that
    has already played, but the resulting numbers are NOT comparable across
    clubs and must not be sorted into a single table.
    """
    if as_of not in ("frozen", "live"):
        raise ValueError(f"as_of must be 'frozen' or 'live', got {as_of!r}")
    bs = bootstrap_static()
    predicts_gw = target_event(bs)
    cutoff = predicts_gw if as_of == "frozen" else None
    players = pd.DataFrame(bs["elements"])
    teams = pd.DataFrame(bs["teams"])[["id", "name", "short_name"]].rename(columns={"id": "team_id"})
    positions = pd.DataFrame(bs["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "element_type", "singular_name_short": "position"}
    )
    players = players.merge(teams, left_on="team", right_on="team_id", how="left")
    players = players.merge(positions, on="element_type", how="left")

    # the FPL API returns most stat fields as strings, not numbers
    numeric_str_cols = [
        "form", "points_per_game", "value_form", "value_season", "ep_next", "ep_this",
        "selected_by_percent", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "ict_index",
        "influence", "creativity", "threat", "price_change_percent",
    ]
    for c in numeric_str_cols:
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors="coerce")

    id_to_team = dict(zip(players["id"], players["name"]))
    id_to_team_id = dict(zip(players["id"], players["team"]))
    gws = fetch_current_gws(players["id"].tolist(), id_to_team, id_to_team_id)
    # debug artifact only -- feat/scored below are built from `gws` in memory,
    # nothing reads this back. Directory may not exist on a fresh clone/deploy
    # (data/raw/ is gitignored, regenerated by collect_history.py locally).
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    gws.to_csv(DATA_DIR / "raw" / "gws_current_live.csv", index=False)

    feat = build_current_features(gws, for_event=cutoff) if not gws.empty else pd.DataFrame()
    for c in FEATURE_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    if not feat.empty:
        # bring position in from `players` (element-summary history has no position column)
        feat = feat.drop(columns=["position"], errors="ignore").merge(
            players[["id", "position"]], left_on="element", right_on="id", how="left"
        )
        feat["position"] = feat["position"].astype("category")

    model = load_model()
    clf = load_minutes_classifier()
    started_reg = load_started_regressor()
    if not feat.empty:
        feat["pred_points"] = model.predict(feat[FEATURE_COLS])
        feat["start_probability"] = clf.predict(feat[FEATURE_COLS]) if clf is not None else np.nan
        feat["pred_points_if_starts"] = (
            started_reg.predict(feat[FEATURE_COLS]) if started_reg is not None else np.nan
        )
    else:
        feat["pred_points"] = np.nan
        feat["start_probability"] = np.nan
        feat["pred_points_if_starts"] = np.nan

    merge_cols = ["element", "pred_points", "start_probability", "pred_points_if_starts", "season_gp_prior"]
    scored = players.merge(
        feat[merge_cols] if not feat.empty else pd.DataFrame(columns=merge_cols),
        left_on="id",
        right_on="element",
        how="left",
    )

    # players with zero current-season history (new signings, no minutes yet,
    # or the model had nothing to score): fall back to a position-median prior.
    fallback = scored["pred_points"].isna()
    median_by_pos = scored.loc[~fallback].groupby("position")["pred_points"].median()
    scored.loc[fallback, "pred_points"] = scored.loc[fallback, "position"].map(median_by_pos)
    median_start_prob_by_pos = scored.loc[~fallback].groupby("position")["start_probability"].median()
    scored.loc[fallback, "start_probability"] = scored.loc[fallback, "position"].map(median_start_prob_by_pos)
    median_if_starts_by_pos = scored.loc[~fallback].groupby("position")["pred_points_if_starts"].median()
    scored.loc[fallback, "pred_points_if_starts"] = scored.loc[fallback, "position"].map(median_if_starts_by_pos)

    scored["now_cost_m"] = scored["now_cost"] / 10.0
    scored["status_ok"] = scored["status"] == "a"

    fdr = next_fixture_summary(n=5, from_event=cutoff)
    scored = scored.merge(fdr, left_on="team_id", right_on="team", how="left")
    scored["next_fdr"] = scored["next_fdr"].fillna(3)
    scored["fdr_next_n_mean"] = scored["fdr_next_n_mean"].fillna(3)
    scored["fdr_multiplier"] = scored["next_fdr"].round().map(FDR_ADJUSTMENT).fillna(1.0)
    scored["pred_points_adj"] = scored["pred_points"] * scored["fdr_multiplier"]
    scored["pred_points_if_starts_adj"] = scored["pred_points_if_starts"] * scored["fdr_multiplier"]

    scored["value_ratio"] = scored["pred_points_adj"] / scored["now_cost_m"]

    # every row records which gameweek it is a prediction FOR, so the UI can
    # never silently sort predictions with different horizons into one list.
    scored["predicts_gw"] = predicts_gw
    scored["scored_as_of"] = as_of

    # price-change signal from FPL's own transfer-momentum projection
    # (likelihood: >=3 imminent rise, <=-3 imminent fall; see bootstrap docs)
    def _likelihood(proj):
        if not isinstance(proj, list) or not proj:
            return 0
        today = next((p for p in proj if p.get("offset") == 0), proj[0])
        return today.get("likelihood", 0)

    scored["price_signal"] = scored["price_change_projections"].apply(_likelihood)
    scored["price_flag"] = scored["price_signal"].apply(
        lambda v: "RISE" if v >= 3 else ("FALL" if v <= -3 else "")
    )

    keep_cols = [
        "id", "web_name", "position", "name", "team_id", "now_cost_m",
        "pred_points", "start_probability", "pred_points_if_starts", "pred_points_if_starts_adj",
        "next_fixture", "next_fdr", "fdr_next_n_mean",
        "pred_points_adj", "value_ratio", "selected_by_percent", "status",
        "status_ok", "chance_of_playing_next_round", "form", "total_points",
        "minutes", "season_gp_prior", "ep_next", "defensive_contribution",
        "tackles", "recoveries", "clearances_blocks_interceptions",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "news", "price_change_percent", "price_signal", "price_flag",
        "predicts_gw", "scored_as_of",
    ]
    return scored[keep_cols].sort_values("pred_points_adj", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    out = score_players()
    out.to_csv(DATA_DIR / "player_predictions.csv", index=False)
    pd.set_option("display.width", 140)
    print(out.head(30).to_string(index=False))
    print(f"\nSaved -> data/player_predictions.csv ({len(out)} players)")
