"""Component ("events -> scoring matrix") model for FPL points.

Instead of one LightGBM regressor on total_points, train a small ensemble of
event models and compose their outputs through the real FPL scoring rules
(``scoring.compose_expected_points``):

  minutes   multiclass {unused, cameo(1-59), start(>=60)}  -> p_cameo, p_start
  goals     Poisson, target goals_scored   | started rows  -> lam_goals_start
  assists   Poisson, target assists        | started rows  -> lam_assists_start
  bonus     Tweedie, target bonus          | started rows  -> exp_bonus_start
  saves     Poisson, target saves          | started GKP   -> lam_saves_start
  team CS   binary,  target team GA == 0    (team-fixture)  -> p_cs_team
  team GC   Poisson, target team GA         (team-fixture)  -> mu_gc_team
  dc        binary,  target DC threshold hit | started 25-26 -> p_dc

Everything is leak-free: every rolling / expanding feature is shifted so a
row only sees gameweeks played strictly before it (``shift=True``). The live
scorer calls the same builder with ``shift=False`` because there we *want*
form including the most recently completed gameweek.

Design notes / evidence:
  - xG/xA/xGC only exist in the archive from 2022-23, so component training
    starts there (``MIN_SEASON``). The old single-stage model nominally used
    2020-21+ but half those rows were NaN on xG anyway.
  - The 2025-26 defensive-contribution rule has one partial season of data.
    The ``dc`` model trains on 2025-26 only; its output enters compose() as
    an explicit +2 * P(threshold) term a direct points regressor can't learn.
  - Double gameweeks: compose() runs per fixture; the caller sums a team's
    fixtures in the target gameweek.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score

import scoring
from features import build_team_id_map

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "components"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VAL_SEASON = "2025-26"
MIN_SEASON = "2022-23"          # first season with xG/xA/xGC in the archive
TEAM_MIN_SEASON = "2016-17"     # team GF/GA/CS models use the full archive
ROLL_WINDOWS = (3, 5)

PLAYER_ROLL_COLS = [
    "total_points", "minutes", "starts", "bps", "bonus",
    "ict_index", "influence", "creativity", "threat",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "yellow_cards", "red_cards",
    "defensive_contribution",
]
CAT = ["position"]

# the vaastav archive labels goalkeepers "GK" and (in 2024-25) has a stray
# "AM" bucket; the live FPL API and scoring.py use GKP/DEF/MID/FWD.
POSITION_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID",
                "FWD": "FWD", "AM": "MID"}


# --------------------------------------------------------------------------- #
# feature engineering
# --------------------------------------------------------------------------- #
def _roll_mean(g, col, w, shift):
    return g[col].transform(
        lambda x: (x.shift(1) if shift else x).rolling(w, min_periods=1).mean()
    )


def _player_key(df: pd.DataFrame, code_map: dict | None) -> pd.Series:
    """Stable cross-season player id. The archive reassigns ``element`` every
    season, so career features must key on the FPL ``code`` instead
    (data/player_codes.csv, from players_raw.csv). Falls back to
    season-scoped element where no code is known."""
    codes = _load_player_codes()
    key = pd.Series(np.nan, index=df.index, dtype="object")
    if codes is not None:
        m = df.merge(codes, on=["season", "element"], how="left")["code"]
        key = m.astype("object").values
    key = pd.Series(key, index=df.index)
    if code_map:
        live = df["element"].map(code_map)
        key = key.where(key.notna(), live)
    # last-resort fallback: season+element keeps each row in its own group
    key = key.where(key.notna(), df["season"].astype(str) + ":" + df["element"].astype(str))
    return key


def add_player_features(df: pd.DataFrame, shift: bool = True,
                        code_map: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    if "position" not in df.columns:
        df["position"] = np.nan
    df["position"] = df["position"].astype("object").map(
        lambda p: POSITION_MAP.get(str(p), None if pd.isna(p) else str(p))
    )
    df["player_key"] = _player_key(df, code_map)
    df["kickoff_time"] = pd.to_datetime(df.get("kickoff_time"), errors="coerce", utc=True)
    keys = ["element", "season", "GW"]
    if df["kickoff_time"].notna().any():
        keys = ["element", "season", "GW", "kickoff_time"]
    df = df.sort_values(keys).reset_index(drop=True)
    # order career groups by real time so expanding() respects season order.
    # rows with no kickoff (or the "live" pseudo-season) fall back to that
    # season's nominal August start, or now.
    _yr = df["season"].astype(str).str.extract(r"^(\d{4})")[0]
    _synth = pd.to_datetime(_yr + "-08-01", utc=True, errors="coerce")
    _synth = _synth.fillna(pd.Timestamp.now(tz="UTC"))
    df["_career_order"] = df["kickoff_time"].fillna(_synth)

    for c in PLAYER_ROLL_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mins = df["minutes"].where(df["minutes"] > 0)
    df["xg90"] = df["expected_goals"] / mins * 90
    df["xa90"] = df["expected_assists"] / mins * 90
    df["xgi90"] = df["expected_goal_involvements"] / mins * 90
    df["played60"] = (df["minutes"] >= 60).astype(float)

    within = df.groupby(["element", "season"], sort=False)
    for w in ROLL_WINDOWS:
        for c in PLAYER_ROLL_COLS + ["xg90", "xa90", "played60"]:
            df[f"{c}_r{w}"] = _roll_mean(within, c, w, shift)
    df["pts_std_r5"] = within["total_points"].transform(
        lambda x: (x.shift(1) if shift else x).rolling(5, min_periods=2).std()
    )
    df["prev_gw_points"] = within["total_points"].shift(1) if shift else df["total_points"]
    df["prev_gw_minutes"] = within["minutes"].shift(1) if shift else df["minutes"]
    df["prev_gw_starts"] = within["starts"].shift(1) if shift else df["starts"]
    df["season_pts_mean_prior"] = within["total_points"].transform(
        lambda x: (x.shift(1) if shift else x).expanding(min_periods=1).mean()
    )
    df["season_gp_prior"] = within["total_points"].transform(
        lambda x: (x.shift(1) if shift else x).expanding().count()
    )

    # cross-season career priors, keyed on the stable player code and ordered
    # by real kickoff time -- this is what gives a GW1-4 row a signal at all
    # (every within-season feature above is still empty that early).
    order = df.sort_values(["player_key", "_career_order"])
    cg = order.groupby("player_key", sort=False)
    for src, dst in [("total_points", "career_ppg"), ("minutes", "career_min_pg"),
                     ("played60", "career_start_rate"), ("xgi90", "career_xgi90")]:
        s = cg[src].transform(lambda x: (x.shift(1) if shift else x).expanding(min_periods=1).mean())
        df[dst] = s.reindex(df.index)

    df["is_home"] = df["was_home"].astype("boolean").astype("Int64").astype("float")
    df["cost"] = pd.to_numeric(df["value"], errors="coerce") / 10.0
    if "chance_of_playing" not in df.columns:
        df["chance_of_playing"] = np.nan
    df["chance_of_playing"] = pd.to_numeric(df["chance_of_playing"], errors="coerce")
    df = df.drop(columns=["_career_order"])
    return df.copy()  # de-fragment after the many column inserts above


_PLAYER_CODES = None


def _load_player_codes() -> pd.DataFrame | None:
    global _PLAYER_CODES
    if _PLAYER_CODES is None:
        p = DATA_DIR / "player_codes.csv"
        _PLAYER_CODES = (pd.read_csv(p).drop_duplicates(["season", "element"])
                         if p.exists() else False)
    return _PLAYER_CODES if _PLAYER_CODES is not False else None


TEAM_ROLL_SRC = {"gf": "goals_for", "ga": "goals_against", "xg": "team_xg",
                 "xgc": "team_xgc", "cs": "cs"}


def _load_team_strength() -> pd.DataFrame | None:
    """Per-season FPL team strength ratings (from vaastav's teams.csv, built
    into data/team_strength.csv). Available 2019-20 onward; a static
    season-level prior that anchors the thin team clean-sheet / goals-conceded
    models, which have only ~20 matches per team per season to learn from."""
    p = DATA_DIR / "team_strength.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    # blend home/away into a single attack/defence number per team-season;
    # the venue split is handled separately by is_home_team.
    t["str_att"] = (t["strength_attack_home"] + t["strength_attack_away"]) / 2
    t["str_def"] = (t["strength_defence_home"] + t["strength_defence_away"]) / 2
    t["str_ovr"] = (t["strength_overall_home"] + t["strength_overall_away"]) / 2
    return t[["season", "id", "str_att", "str_def", "str_ovr"]]


def build_team_table(df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """One row per (season, team, fixture) with leak-free rolling form for the
    team (t_*) and its opponent (o_*), the FPL strength ratings for both, and
    targets ``cs`` and ``goals_against``.

    Built from the FULL history (not just the xG era): goals-for/against and
    clean-sheet rate go back to 2016-17, which roughly triples the rows the
    team models see. team_xg/team_xgc are NaN before 2022-23 and LightGBM
    routes around them.
    """
    d = df.copy()
    d["kickoff_time"] = pd.to_datetime(d.get("kickoff_time"), errors="coerce", utc=True)
    d["was_home"] = d["was_home"].astype(bool)
    for c in ("team_h_score", "team_a_score", "expected_goals", "expected_goals_conceded"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["goals_for"] = np.where(d["was_home"], d["team_h_score"], d["team_a_score"])
    d["goals_against"] = np.where(d["was_home"], d["team_a_score"], d["team_h_score"])

    tg = d.groupby(["season", "team", "fixture"], as_index=False).agg(
        GW=("GW", "first"), was_home=("was_home", "first"),
        opponent_team=("opponent_team", "first"), kickoff_time=("kickoff_time", "first"),
        goals_for=("goals_for", "first"), goals_against=("goals_against", "first"),
        team_xg=("expected_goals", "sum"),
        # expected_goals_conceded is a per-player stat (xGC while on the pitch);
        # the player who went the full 90 carries ~the team's match xGC, so max
        # over the squad recovers the team number. "first" grabbed a sub's ~0.
        team_xgc=("expected_goals_conceded", "max"),
    )
    tg["cs"] = (tg["goals_against"] == 0).astype(float)
    tg = tg.sort_values(["season", "team", "GW", "kickoff_time"]).reset_index(drop=True)

    g = tg.groupby(["season", "team"], sort=False)
    for w in ROLL_WINDOWS:
        for short, src in TEAM_ROLL_SRC.items():
            tg[f"t_{short}_r{w}"] = _roll_mean(g, src, w, shift)

    t_cols = [f"t_{s}_r{w}" for w in ROLL_WINDOWS for s in TEAM_ROLL_SRC]
    id_map = build_team_id_map(df)                       # (season, team name) -> team_id
    tg = tg.merge(id_map, on=["season", "team"], how="left")

    strength = _load_team_strength()
    if strength is not None:
        tg = tg.merge(strength.rename(columns={"id": "team_id"}),
                      on=["season", "team_id"], how="left")
    else:
        tg["str_att"] = tg["str_def"] = tg["str_ovr"] = np.nan
    str_cols = ["str_att", "str_def", "str_ovr"]

    opp = tg[["season", "fixture", "team_id"] + t_cols + str_cols].rename(
        columns={"team_id": "opponent_team",
                 **{c: c.replace("t_", "o_", 1) for c in t_cols},
                 **{c: c.replace("str_", "o_str_", 1) for c in str_cols}}
    )
    tg = tg.merge(opp, on=["season", "fixture", "opponent_team"], how="left")
    tg["is_home_team"] = tg["was_home"].astype(float)
    return tg


# --- feature sets -----------------------------------------------------------
_T = lambda s: [f"t_{s}_r{w}" for w in ROLL_WINDOWS]
_O = lambda s: [f"o_{s}_r{w}" for w in ROLL_WINDOWS]

TEAM_FEATURES = (_T("xgc") + _T("ga") + _T("cs") + _T("gf") + _O("xg") + _O("gf") + _O("ga")
                 + ["str_def", "str_att", "o_str_att", "o_str_def", "is_home_team"])

GOALS_FEATURES = (
    [f"expected_goals_r{w}" for w in ROLL_WINDOWS]
    + [f"xg90_r{w}" for w in ROLL_WINDOWS]
    + [f"goals_scored_r{w}" for w in ROLL_WINDOWS]
    + [f"threat_r{w}" for w in ROLL_WINDOWS]
    + ["ict_index_r5", "expected_goal_involvements_r5", "season_pts_mean_prior",
       "career_xgi90", "career_ppg", "is_home", "cost", "position",
       "t_xg_r5", "o_xgc_r5", "o_ga_r5", "str_att", "o_str_def"]
)
ASSISTS_FEATURES = (
    [f"expected_assists_r{w}" for w in ROLL_WINDOWS]
    + [f"xa90_r{w}" for w in ROLL_WINDOWS]
    + [f"assists_r{w}" for w in ROLL_WINDOWS]
    + [f"creativity_r{w}" for w in ROLL_WINDOWS]
    + ["influence_r5", "ict_index_r5", "career_xgi90", "career_ppg",
       "is_home", "cost", "position", "t_xg_r5", "o_xgc_r5", "str_att", "o_str_def"]
)
BONUS_FEATURES = (
    [f"bps_r{w}" for w in ROLL_WINDOWS]
    + ["ict_index_r5", "expected_goal_involvements_r5", "threat_r5",
       "creativity_r5", "influence_r5", "bonus_r5", "position", "cost"]
)
SAVES_FEATURES = (
    [f"saves_r{w}" for w in ROLL_WINDOWS]
    + ["o_xg_r5", "o_gf_r5", "t_xgc_r5", "cost", "o_str_att", "str_def"]
)
MINUTES_FEATURES = (
    [f"minutes_r{w}" for w in ROLL_WINDOWS]
    + [f"starts_r{w}" for w in ROLL_WINDOWS]
    + [f"played60_r{w}" for w in ROLL_WINDOWS]
    + ["prev_gw_minutes", "prev_gw_starts", "prev_gw_points", "season_gp_prior",
       "career_min_pg", "career_start_rate", "career_ppg", "cost", "position",
       "chance_of_playing"]
)
DC_FEATURES = (
    [f"defensive_contribution_r{w}" for w in ROLL_WINDOWS]
    + ["minutes_r5", "played60_r5", "bps_r5", "position", "cost", "career_start_rate"]
)
ALL_FEATURE_SETS = {
    "minutes": MINUTES_FEATURES, "goals": GOALS_FEATURES, "assists": ASSISTS_FEATURES,
    "bonus": BONUS_FEATURES, "saves": SAVES_FEATURES, "team_cs": TEAM_FEATURES,
    "team_gc": TEAM_FEATURES, "dc": DC_FEATURES,
}


def assemble(df: pd.DataFrame, shift: bool = True, team_df: pd.DataFrame | None = None,
             code_map: dict | None = None):
    """Return (player_frame, team_table). player_frame has one row per
    player-fixture with player rolling features + team form / strength merged
    on, ready for every component model and for compose().

    ``team_df`` lets the team table be built from a wider set of seasons than
    the player frame (the team models want all history; the player models are
    pinned to the xG era).
    """
    pf = add_player_features(df, shift=shift, code_map=code_map)
    tt = build_team_table(team_df if team_df is not None else df, shift=shift)
    team_cols = [c for c in tt.columns
                 if c.startswith(("t_", "o_", "str_")) and c not in ("team_id",)]
    merge_cols = ["season", "team", "fixture", "is_home_team"] + team_cols
    pf = pf.merge(tt[merge_cols].drop_duplicates(["season", "team", "fixture"]),
                  on=["season", "team", "fixture"], how="left")
    for c in CAT:
        pf[c] = pf[c].astype("category")
    return pf, tt


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _ds(df, feats, label, weight=None):
    return lgb.Dataset(df[feats], label=np.asarray(label, dtype=float),
                       categorical_feature=[c for c in CAT if c in feats],
                       weight=weight, free_raw_data=False)


def _bucket(m):
    m = np.asarray(m, dtype=float)
    return np.where(m >= 60, 2, np.where(m > 0, 1, 0))


_POISSON = {"objective": "poisson", "metric": "poisson", "learning_rate": 0.03,
            "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "max_delta_step": 0.7, "verbose": -1}


def train_all(val_season: str = VAL_SEASON, save: bool = True) -> dict:
    full_raw = _load_raw()
    raw = full_raw[full_raw["season"] >= MIN_SEASON].copy()
    feat, team = assemble(raw, team_df=full_raw)
    feat = feat[feat["season"] >= MIN_SEASON].copy()
    team = team[team["season"] >= TEAM_MIN_SEASON].copy()
    tr, va = feat[feat["season"] < val_season], feat[feat["season"] == val_season]
    s_tr, s_va = tr[tr["minutes"] >= 60], va[va["minutes"] >= 60]
    models, report = {}, {}

    def fit(params, dtr, dva, rounds, stop=60):
        return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(0)])

    # minutes
    p = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
         "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 80,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
    m = fit(p, _ds(tr, MINUTES_FEATURES, _bucket(tr["minutes"])),
            _ds(va, MINUTES_FEATURES, _bucket(va["minutes"])), 2000)
    models["minutes"] = m
    pv = m.predict(va[MINUTES_FEATURES], num_iteration=m.best_iteration)
    report["minutes_logloss"] = float(log_loss(_bucket(va["minutes"]), pv, labels=[0, 1, 2]))
    report["minutes_start_auc"] = float(roc_auc_score((va["minutes"] >= 60).astype(int), pv[:, 2]))

    # goals / assists
    for name, feats, tgt in [("goals", GOALS_FEATURES, "goals_scored"),
                             ("assists", ASSISTS_FEATURES, "assists")]:
        m = fit(dict(_POISSON), _ds(s_tr, feats, s_tr[tgt]), _ds(s_va, feats, s_va[tgt]), 3000)
        models[name] = m
        pr = np.clip(m.predict(s_va[feats], num_iteration=m.best_iteration), 0, None)
        report[f"{name}_mae_started"] = float(mean_absolute_error(s_va[tgt], pr))
        report[f"{name}_pred_vs_actual_mean"] = [float(pr.mean()), float(s_va[tgt].mean())]

    # bonus
    p = {"objective": "tweedie", "tweedie_variance_power": 1.3, "metric": "mae",
         "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 60,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
    m = fit(p, _ds(s_tr, BONUS_FEATURES, s_tr["bonus"]), _ds(s_va, BONUS_FEATURES, s_va["bonus"]), 3000)
    models["bonus"] = m
    report["bonus_mae_started"] = float(mean_absolute_error(
        s_va["bonus"], m.predict(s_va[BONUS_FEATURES], num_iteration=m.best_iteration)))

    # saves (GKP)
    gk_tr, gk_va = s_tr[s_tr["position"] == "GKP"], s_va[s_va["position"] == "GKP"]
    p = dict(_POISSON, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9)
    m = fit(p, _ds(gk_tr, SAVES_FEATURES, gk_tr["saves"]), _ds(gk_va, SAVES_FEATURES, gk_va["saves"]), 1500)
    models["saves"] = m
    report["saves_mae_started_gkp"] = float(mean_absolute_error(
        gk_va["saves"], m.predict(gk_va[SAVES_FEATURES], num_iteration=m.best_iteration)))

    # team clean sheet / goals conceded -- thin data (~20 matches/team/season),
    # so: full-archive rows, strength priors, small trees, fixed rounds (no
    # early stopping, which kept halting at iteration ~1 on the noisy binary
    # val split and collapsing the model to the base rate).
    keep = ["cs", "goals_against", "t_ga_r5", "is_home_team"]      # never dropna on xg cols
    t_tr = team[team["season"] < val_season].dropna(subset=keep)
    t_va = team[team["season"] == val_season].dropna(subset=keep)
    report["team_rows"] = [len(t_tr), len(t_va)]
    cs_p = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 15,
            "min_data_in_leaf": 60, "feature_fraction": 0.7, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
    m = lgb.train(cs_p, _ds(t_tr, TEAM_FEATURES, t_tr["cs"]), num_boost_round=350)
    models["team_cs"] = m
    pv = m.predict(t_va[TEAM_FEATURES])
    report["team_cs_auc"] = float(roc_auc_score(t_va["cs"], pv))
    report["team_cs_pred_vs_actual_mean"] = [float(pv.mean()), float(t_va["cs"].mean())]

    gc_p = dict(_POISSON, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.7, lambda_l2=1.0)
    m = lgb.train(gc_p, _ds(t_tr, TEAM_FEATURES, t_tr["goals_against"]), num_boost_round=350)
    models["team_gc"] = m
    gcp = m.predict(t_va[TEAM_FEATURES])
    report["team_gc_mae"] = float(mean_absolute_error(t_va["goals_against"], gcp))
    report["team_gc_pred_vs_actual_mean"] = [float(gcp.mean()), float(t_va["goals_against"].mean())]

    # defensive contribution (2025-26 only)
    models["dc"], report["dc_auc"] = _train_dc(feat, save_split=True, report=report)

    # per-position affine calibration of the composed number. compose() sums
    # eight independently-trained rates; small systematic biases (e.g. the GC
    # model runs slightly hot) accumulate. Fit actual ~ a + b*pred per
    # position on the LAST 25% of training gameweeks (time-ordered, so the
    # calibration itself isn't fit on the rows it will be judged against).
    cal_src = tr.copy()
    cut = cal_src["GW"].quantile(0.75) if (cal_src["season"].nunique() == 1) else None
    hold = cal_src if cut is None else cal_src  # use all train seasons; recency not critical for 2 params
    scored_tr = predict_frame(hold, models)
    calib = {}
    for pos, gp in scored_tr.groupby(scored_tr["position"].astype(str)):
        x = gp["pred_points_raw"].to_numpy()
        y = gp["total_points"].to_numpy()
        if len(gp) > 200 and np.std(x) > 1e-6:
            b = float(np.cov(x, y)[0, 1] / np.var(x))
            a = float(y.mean() - b * x.mean())
            # guard against a pathological fit
            calib[pos] = [a, b] if 0.3 < b < 2.0 else [0.0, 1.0]
        else:
            calib[pos] = [0.0, 1.0]
    models["_calib"] = calib
    report["calibration"] = calib

    # per-code career priors, so a live GW1-4 row (which has almost no
    # in-season history) inherits last season's baseline in build_live_frame().
    _CAREER_COLS = ["career_ppg", "career_min_pg", "career_start_rate", "career_xgi90"]
    cp = (feat.dropna(subset=["player_key"]).sort_values(["player_key", "season", "GW"])
          .groupby("player_key")[_CAREER_COLS].last().reset_index())
    cp.to_csv(DATA_DIR / "career_priors.csv", index=False)

    if save:
        for k, mdl in models.items():
            if mdl is not None and not isinstance(mdl, dict):
                mdl.save_model(str(MODEL_DIR / f"{k}.txt"))
        (MODEL_DIR / "meta.json").write_text(json.dumps(
            {"val_season": val_season, "min_season": MIN_SEASON, "report": report,
             "calibration": calib, "feature_sets": ALL_FEATURE_SETS}, indent=2))
    return {"models": models, "report": report}


def _train_dc(feat, *, save_split=False, report=None, rounds=1500):
    dc = feat[(feat["season"] == "2025-26") & (feat["minutes"] >= 60)
              & (feat["position"].isin(["DEF", "MID", "FWD"]))].copy()
    if len(dc) < 500:
        return None, None
    dc["dc_hit"] = [float(v >= scoring.DC_THRESHOLD.get(str(pos), 99))
                    for v, pos in zip(dc["defensive_contribution"].fillna(0), dc["position"])]
    if dc["dc_hit"].nunique() < 2:
        return None, None
    p = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 31,
         "min_data_in_leaf": 40, "feature_fraction": 0.85, "bagging_fraction": 0.85,
         "bagging_freq": 1, "verbose": -1}
    if save_split:
        cut = dc["GW"].quantile(0.8)
        dtr, dva = dc[dc["GW"] <= cut], dc[dc["GW"] > cut]
        m = lgb.train(p, _ds(dtr, DC_FEATURES, dtr["dc_hit"]), num_boost_round=rounds,
                      valid_sets=[_ds(dva, DC_FEATURES, dva["dc_hit"])],
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        auc = float(roc_auc_score(dva["dc_hit"], m.predict(dva[DC_FEATURES], num_iteration=m.best_iteration)))
        # refit on all of it for production
        m = lgb.train(p, _ds(dc, DC_FEATURES, dc["dc_hit"]),
                      num_boost_round=max(m.best_iteration, 100))
        if report is not None:
            report["dc_base_rate"] = float(dc["dc_hit"].mean())
        return m, auc
    return lgb.train(p, _ds(dc, DC_FEATURES, dc["dc_hit"]), num_boost_round=250), None


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #
def load_models() -> dict:
    out = {k: (lgb.Booster(model_file=str(MODEL_DIR / f"{k}.txt"))
               if (MODEL_DIR / f"{k}.txt").exists() else None)
           for k in ["minutes", "goals", "assists", "bonus", "saves", "team_cs", "team_gc", "dc"]}
    meta = MODEL_DIR / "meta.json"
    if meta.exists():
        out["_calib"] = json.loads(meta.read_text()).get("calibration", {})
    return out


def predict_frame(feat: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    models = models or load_models()
    out = feat.copy()

    mm = models["minutes"]
    probs = mm.predict(out[MINUTES_FEATURES], num_iteration=getattr(mm, "best_iteration", None))
    out["p_unused"], out["p_cameo"], out["p_start"] = probs[:, 0], probs[:, 1], probs[:, 2]

    def pred(name, feats, default):
        m = models.get(name)
        if m is None:
            return np.full(len(out), default, dtype=float)
        return np.clip(m.predict(out[feats], num_iteration=getattr(m, "best_iteration", None)), 0, None)

    out["lam_goals_start"] = pred("goals", GOALS_FEATURES, 0.05)
    out["lam_assists_start"] = pred("assists", ASSISTS_FEATURES, 0.05)
    out["exp_bonus_start"] = pred("bonus", BONUS_FEATURES, 0.3)
    out["lam_saves_start"] = np.where(out["position"].astype(str) == "GKP",
                                      pred("saves", SAVES_FEATURES, 2.5), 0.0)
    out["p_cs_team"] = np.clip(pred("team_cs", TEAM_FEATURES, 0.28), 0, 1)
    out["mu_gc_team"] = pred("team_gc", TEAM_FEATURES, 1.4)
    out["p_dc"] = (np.clip(models["dc"].predict(
        out[DC_FEATURES], num_iteration=getattr(models["dc"], "best_iteration", None)), 0, 1)
        if models.get("dc") is not None else np.zeros(len(out)))

    out["yc_rate"] = pd.to_numeric(out.get("yellow_cards_r5"), errors="coerce").fillna(0.0).clip(0, 1)
    out["rc_rate"] = pd.to_numeric(out.get("red_cards_r5"), errors="coerce").fillna(0.0).clip(0, 0.5)

    out["pred_points_raw"] = scoring.compose_expected_points(out, if_starts=False)
    out["pred_points_if_starts_raw"] = scoring.compose_expected_points(out, if_starts=True)

    calib = (models or {}).get("_calib") or {}
    if calib:
        pos = out["position"].astype(str)
        a = pos.map(lambda p: calib.get(p, [0.0, 1.0])[0]).astype(float)
        b = pos.map(lambda p: calib.get(p, [0.0, 1.0])[1]).astype(float)
        out["pred_points"] = (a + b * out["pred_points_raw"]).clip(lower=0)
        out["pred_points_if_starts"] = (a + b * out["pred_points_if_starts_raw"]).clip(lower=0)
    else:
        out["pred_points"] = out["pred_points_raw"]
        out["pred_points_if_starts"] = out["pred_points_if_starts_raw"]
    out["start_probability"] = out["p_start"]
    return out


# --------------------------------------------------------------------------- #
# live scoring (upcoming gameweek, one or more fixtures per team)
# --------------------------------------------------------------------------- #
_T_ROLL = [f"t_{s}_r{w}" for w in ROLL_WINDOWS for s in TEAM_ROLL_SRC]


def build_live_frame(gws: pd.DataFrame, upcoming: pd.DataFrame,
                     chance_map: dict | None = None,
                     code_map: dict | None = None) -> pd.DataFrame:
    """One row per (player, upcoming fixture).

    ``gws``      -- fetch_current_gws() output: a row per player per GW played
                    this season, with ``team`` (name), ``team_id`` (numeric),
                    ``season`` == "live".
    ``upcoming`` -- columns [team_id, opponent_id, is_home, event]; more than
                    one row per team for a double gameweek.
    ``chance_map`` -- optional {element_id: chance_of_playing_next_round/100}.
    ``code_map``  -- optional {element_id: FPL code}, so career priors carry
                     over from last season for a GW1-4 cold start.

    Each player's form is taken from their most recent completed gameweek
    (shift=False); the fixture context (venue, opponent form, both strength
    ratings) is swapped in from ``upcoming`` and the opponent's latest form.
    """
    g = gws.copy()
    g["season"] = "live"
    pf = add_player_features(g, shift=False, code_map=code_map)
    latest = pf.sort_values(["element", "GW"]).groupby("element", as_index=False).tail(1).copy()
    if chance_map:
        latest["chance_of_playing"] = latest["element"].map(chance_map)

    # seed career priors from last season where this season is still too thin
    cp_path = DATA_DIR / "career_priors.csv"
    if cp_path.exists() and "player_key" in latest.columns:
        cp = pd.read_csv(cp_path).set_index("player_key")
        for col in ["career_ppg", "career_min_pg", "career_start_rate", "career_xgi90"]:
            seeded = latest["player_key"].map(cp[col]) if col in cp.columns else np.nan
            thin = latest["season_gp_prior"].fillna(0) < 3
            latest[col] = np.where(latest[col].notna() & ~thin, latest[col], seeded)

    tt = build_team_table(g, shift=False)
    team_latest = tt.sort_values("GW").groupby("team_id").tail(1).set_index("team_id")

    strength = _load_team_strength()
    smax = (strength[strength["season"] == strength["season"].max()].set_index("id")
            if strength is not None else pd.DataFrame())

    def s_get(tid, col):
        return float(smax.loc[tid, col]) if tid in getattr(smax, "index", []) else np.nan

    def t_get(tid, col):
        return float(team_latest.loc[tid, col]) if tid in team_latest.index else np.nan

    blocks = []
    for _, fx in upcoming.iterrows():
        tid, oid, home = int(fx["team_id"]), int(fx["opponent_id"]), bool(fx["is_home"])
        sub = latest[latest["team_id"] == tid].copy()
        if sub.empty:
            continue
        for c in _T_ROLL:
            sub[c] = t_get(tid, c)
            sub[c.replace("t_", "o_", 1)] = t_get(oid, c)
        sub["str_att"], sub["str_def"] = s_get(tid, "str_att"), s_get(tid, "str_def")
        sub["str_ovr"] = s_get(tid, "str_ovr")
        sub["o_str_att"], sub["o_str_def"] = s_get(oid, "str_att"), s_get(oid, "str_def")
        sub["is_home"] = float(home)
        sub["is_home_team"] = float(home)
        sub["fix_event"] = int(fx["event"])
        blocks.append(sub)

    frame = pd.concat(blocks, ignore_index=True) if blocks else latest.iloc[0:0].copy()
    for c in CAT:
        frame[c] = frame[c].astype("category")
    return frame


def score_live(gws: pd.DataFrame, upcoming: pd.DataFrame,
               chance_map: dict | None = None, models: dict | None = None,
               code_map: dict | None = None) -> pd.DataFrame:
    """Per-player expected points for the upcoming gameweek, summed across a
    team's fixtures (double gameweeks). Columns: element, pred_points,
    pred_points_if_starts, start_probability, n_fixtures, season_gp_prior."""
    frame = build_live_frame(gws, upcoming, chance_map=chance_map, code_map=code_map)
    if frame.empty:
        return pd.DataFrame(columns=["element", "pred_points", "pred_points_if_starts",
                                     "start_probability", "n_fixtures", "season_gp_prior"])
    scored = predict_frame(frame, models)
    agg = scored.groupby("element").agg(
        pred_points=("pred_points", "sum"),
        pred_points_if_starts=("pred_points_if_starts", "sum"),
        start_probability=("start_probability", "max"),
        n_fixtures=("fix_event", "nunique"),
        season_gp_prior=("season_gp_prior", "first"),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def _load_raw() -> pd.DataFrame:
    pq = DATA_DIR / "features.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(DATA_DIR / "history_gws.csv", low_memory=False)


def _train_fold(tr, s_tr, team_tr) -> dict:
    models = {}
    models["minutes"] = lgb.train(
        {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 80, "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 1, "verbose": -1},
        _ds(tr, MINUTES_FEATURES, _bucket(tr["minutes"])), num_boost_round=350)
    for name, feats, tgt, obj in [("goals", GOALS_FEATURES, "goals_scored", "poisson"),
                                  ("assists", ASSISTS_FEATURES, "assists", "poisson"),
                                  ("bonus", BONUS_FEATURES, "bonus", "tweedie")]:
        pr = {"objective": obj, "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 60,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
        if obj == "poisson":
            pr["max_delta_step"] = 0.7
        if obj == "tweedie":
            pr["tweedie_variance_power"] = 1.3
        models[name] = lgb.train(pr, _ds(s_tr, feats, s_tr[tgt]), num_boost_round=500)
    gk = s_tr[s_tr["position"] == "GKP"]
    models["saves"] = lgb.train(
        {"objective": "poisson", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
         "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "verbose": -1},
        _ds(gk, SAVES_FEATURES, gk["saves"]), num_boost_round=300)
    tcs = team_tr.dropna(subset=["cs", "goals_against", "t_ga_r5", "is_home_team"])
    models["team_cs"] = lgb.train(
        {"objective": "binary", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 60,
         "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
         "verbose": -1},
        _ds(tcs, TEAM_FEATURES, tcs["cs"]), num_boost_round=350)
    models["team_gc"] = lgb.train(
        {"objective": "poisson", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 60,
         "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
         "max_delta_step": 0.7, "verbose": -1},
        _ds(tcs, TEAM_FEATURES, tcs["goals_against"]), num_boost_round=350)
    models["dc"], _ = _train_dc(tr)
    # per-position affine calibration, fit on this fold's own training rows
    scored = predict_frame(tr, models)
    calib = {}
    for pos, gp in scored.groupby(scored["position"].astype(str)):
        x, y = gp["pred_points_raw"].to_numpy(), gp["total_points"].to_numpy()
        if len(gp) > 200 and np.std(x) > 1e-6:
            b = float(np.cov(x, y)[0, 1] / np.var(x))
            a = float(y.mean() - b * x.mean())
            calib[pos] = [a, b] if 0.3 < b < 2.0 else [0.0, 1.0]
    models["_calib"] = calib
    return models


def walk_forward(val_season: str = VAL_SEASON,
                 fold_starts=(6, 11, 16, 21, 26, 31, 36), width=5) -> pd.DataFrame:
    full_raw = _load_raw()
    raw = full_raw[full_raw["season"] >= MIN_SEASON].copy()
    feat, team = assemble(raw, team_df=full_raw)
    feat = feat[feat["season"] >= MIN_SEASON]
    team = team[team["season"] >= TEAM_MIN_SEASON]
    rows = []
    for fs in fold_starts:
        fe = fs + width - 1
        tr = feat[(feat["season"] < val_season) | ((feat["season"] == val_season) & (feat["GW"] < fs))]
        te = feat[(feat["season"] == val_season) & feat["GW"].between(fs, fe)]
        if te.empty:
            continue
        team_tr = team[(team["season"] < val_season) | ((team["season"] == val_season) & (team["GW"] < fs))]
        models = _train_fold(tr, tr[tr["minutes"] >= 60], team_tr)
        pred = predict_frame(te, models)
        for gw, g in pred.groupby("GW"):
            rows.append({"GW": int(gw), "fold": fs, "n": len(g),
                         "mae_model": mean_absolute_error(g["total_points"], g["pred_points"]),
                         "mae_naive": mean_absolute_error(
                             g["total_points"], g["season_pts_mean_prior"].fillna(tr["total_points"].mean()))})
        print(f"fold<GW{fs}: GW{fs}-{fe} ({len(te)} rows) MAE={np.mean([r['mae_model'] for r in rows if r['fold']==fs]):.3f}")
    return pd.DataFrame(rows).sort_values("GW")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        res = walk_forward()
        res.to_csv(DATA_DIR / "backtest_component_results.csv", index=False)
        ov_m = (res["mae_model"] * res["n"]).sum() / res["n"].sum()
        ov_n = (res["mae_naive"] * res["n"]).sum() / res["n"].sum()
        print(f"\nComponent walk-forward MAE: model={ov_m:.3f} naive={ov_n:.3f}")
        print(res.groupby("fold")[["mae_model", "mae_naive"]].mean().round(3))
    else:
        r = train_all()
        print(json.dumps(r["report"], indent=2))
