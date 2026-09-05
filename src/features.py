"""Feature engineering for the FPL points-prediction model.

All rolling/lag features are shifted so that a row's features only ever use
information available strictly BEFORE that gameweek was played (no leakage).
"""
import numpy as np
import pandas as pd

DATA_DIR_HIST = "data/history_gws.csv"

ROLL_WINDOWS = (3, 5)

NUMERIC_ROLL_COLS = [
    "total_points",
    "minutes",
    "bps",
    "ict_index",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "goals_scored",
    "assists",
    "clean_sheets",
]


def _team_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team, GW): goals scored/conceded that gameweek,
    used to build opponent-strength features."""
    df = df.copy()
    df["goals_for"] = np.where(df.was_home, df.team_h_score, df.team_a_score)
    df["goals_against"] = np.where(df.was_home, df.team_a_score, df.team_h_score)
    tg = (
        df.groupby(["season", "team", "GW"], as_index=False)
        .agg(goals_for=("goals_for", "first"), goals_against=("goals_against", "first"))
    )
    tg = tg.sort_values(["season", "team", "GW"])
    for w in ROLL_WINDOWS:
        tg[f"team_gf_{w}"] = (
            tg.groupby(["season", "team"])["goals_for"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        tg[f"team_ga_{w}"] = (
            tg.groupby(["season", "team"])["goals_against"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
    return tg


def build_team_id_map(df: pd.DataFrame) -> pd.DataFrame:
    """(season, team NAME) -> numeric team id, recovered purely from the data:
    each row already links its own team's NAME to the OPPONENT's numeric id
    (`opponent_team`). For a given (season, fixture) there are exactly two
    sides; self-joining on (season, fixture) and keeping the cross pair
    recovers each side's own numeric id from the other side's declared
    opponent -- no external team-id table needed, and it's robust to a
    team's id changing between seasons since it's rebuilt from that season's
    own fixtures every time.
    """
    pairs = df[["season", "fixture", "team", "opponent_team"]].drop_duplicates()
    merged = pairs.merge(pairs, on=["season", "fixture"], suffixes=("", "_other"))
    merged = merged[merged["team"] != merged["team_other"]]
    id_map = (
        merged[["season", "team", "opponent_team_other"]]
        .drop_duplicates()
        .rename(columns={"opponent_team_other": "team_id"})
    )
    return id_map


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["element", "season", "GW"]).reset_index(drop=True)

    for col in NUMERIC_ROLL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    grp = df.groupby(["element", "season"])
    for w in ROLL_WINDOWS:
        for col in NUMERIC_ROLL_COLS:
            df[f"{col}_r{w}"] = grp[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # career-to-date (within-season) form, wider signal than short rolling windows
    df["season_pts_mean_prior"] = grp["total_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["season_gp_prior"] = grp["total_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )
    df["prev_gw_points"] = grp["total_points"].shift(1)
    df["prev_gw_minutes"] = grp["minutes"].shift(1)

    # own-team recent scoring/defensive form (proxy for fixture context)
    tg = _team_game_table(df)
    opp_cols = [c for c in tg.columns if c.startswith("team_g")]
    df = df.merge(
        tg.rename(columns={c: f"self_{c}" for c in opp_cols}),
        on=["season", "team", "GW"],
        how="left",
    )

    # opponent's recent scoring/defensive form -- the actual fixture-difficulty
    # signal, trained directly into the model rather than applied as a
    # post-hoc multiplier on top of predictions.
    id_map = build_team_id_map(df)
    tg_with_id = tg.merge(id_map, on=["season", "team"], how="left")
    opp_join = tg_with_id[["season", "team_id", "GW"] + opp_cols].rename(
        columns={"team_id": "opponent_team", **{c: f"opp_{c}" for c in opp_cols}}
    )
    df = df.merge(opp_join, on=["season", "opponent_team", "GW"], how="left")

    null_frac = df["opp_team_gf_3"].isna().mean()
    if null_frac > 0.05:
        print(f"WARNING: opponent-form merge left {null_frac:.1%} of rows null "
              f"(expected near-0 outside each player's first {ROLL_WINDOWS[0]} games)")

    df["is_home"] = df["was_home"].astype("boolean").astype("Int64")
    df["cost"] = df["value"] / 10.0

    return df


FEATURE_COLS = (
    [f"{c}_r{w}" for w in ROLL_WINDOWS for c in NUMERIC_ROLL_COLS]
    + ["season_pts_mean_prior", "season_gp_prior", "prev_gw_points", "prev_gw_minutes"]
    + [f"self_team_gf_{w}" for w in ROLL_WINDOWS]
    + [f"self_team_ga_{w}" for w in ROLL_WINDOWS]
    + [f"opp_team_gf_{w}" for w in ROLL_WINDOWS]
    + [f"opp_team_ga_{w}" for w in ROLL_WINDOWS]
    + ["is_home", "cost", "position"]
)

TARGET_COL = "total_points"


if __name__ == "__main__":
    raw = pd.read_csv(DATA_DIR_HIST, low_memory=False)
    feat = build_feature_table(raw)

    # float32 for the numeric feature columns roughly halves memory/disk vs
    # float64 -- plenty of precision for rolling averages of small integers.
    float_cols = [c for c in FEATURE_COLS if c != "position" and feat[c].dtype.kind == "f"]
    feat[float_cols] = feat[float_cols].astype("float32")

    feat.to_parquet("data/features.parquet", index=False)
    print(f"Saved -> data/features.parquet ({len(feat)} rows, "
          f"{feat.memory_usage(deep=True).sum() / 1e6:.1f}MB in memory)")
    print(feat[FEATURE_COLS + [TARGET_COL]].describe(include="all").T)
