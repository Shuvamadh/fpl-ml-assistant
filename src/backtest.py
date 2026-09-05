"""Proper walk-forward validation: simulates exactly how the model gets used
in production -- train on everything known SO FAR, predict the next block of
gameweeks, never look forward. This is the honest test of "how accurate would
these predictions actually have been, gameweek by gameweek, through a real
season" -- as opposed to train_model.py's single train/holdout-season split,
which only tells you the model generalises to an unseen SEASON, not how it
performs as a season unfolds gameweek by gameweek.

Also demonstrates the two training regimes the model actually needs to
support in production:
  (a) cross-season generalisation: train on seasons 2016-17..2024-25 (as much
      history as exists), predict an entirely unseen season (2025-26) -- this
      is what train_model.py already does.
  (b) within-season walk-forward: as 2025-26 (or the live 2026-27 season)
      progresses gameweek by gameweek, keep retraining on everything seen so
      far (all history + this season's completed gameweeks) and predict the
      next block -- this is what predict.py effectively relies on via the
      model's season-to-date features, and what this script explicitly
      backtests to quantify.
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

from features import FEATURE_COLS, TARGET_COL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CAT_COLS = ["position"]

BACKTEST_SEASON = "2025-26"  # most recent COMPLETE season -- real outcomes known for all 38 GWs
FOLD_STARTS = [6, 11, 16, 21, 26, 31, 36]  # expanding-window checkpoints within that season
FOLD_WIDTH = 5  # predict this many gameweeks per fold before retraining

# seasons before this lack position/team/xG tracking in the archive -- same
# cutoff as train_model.py's MIN_SEASON, kept in sync manually since the two
# scripts intentionally don't share a config module.
MIN_SEASON = "2020-21"
# rolling-origin validation seasons: is MAE~0.96 stable, or one lucky draw?
ROLLING_ORIGIN_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}
NUM_ROUNDS = 400  # fixed (no early stopping) to keep backtest folds fast and comparable


def load_features(min_season: str | None = MIN_SEASON) -> pd.DataFrame:
    needed = list(dict.fromkeys(FEATURE_COLS + [TARGET_COL, "season", "season_gp_prior", "GW"]))
    df = pd.read_parquet(DATA_DIR / "features.parquet", columns=needed)
    mask = df["season_gp_prior"] >= 1
    if min_season:
        mask &= df["season"] >= min_season
    df = df[mask].copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def rolling_origin_cv(seasons: list[str] = ROLLING_ORIGIN_SEASONS) -> pd.DataFrame:
    """For each of the last few seasons, train ONLY on strictly-earlier
    seasons (respecting MIN_SEASON) and validate on it. Reveals whether the
    single-holdout MAE (~0.96) is a stable number or a lucky/unlucky draw for
    that one season -- a spread across folds answers that a single number
    can't."""
    df = load_features()
    rows = []
    for season in seasons:
        train_df = df[df["season"] < season]
        val_df = df[df["season"] == season]
        if train_df.empty or val_df.empty:
            print(f"skip {season}: no training data strictly before it "
                  f"(MIN_SEASON={MIN_SEASON})")
            continue
        train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df[TARGET_COL], categorical_feature=CAT_COLS)
        model = lgb.train(PARAMS, train_set, num_boost_round=NUM_ROUNDS)
        preds = model.predict(val_df[FEATURE_COLS])
        mae = mean_absolute_error(val_df[TARGET_COL], preds)
        naive = mean_absolute_error(val_df[TARGET_COL], val_df["season_pts_mean_prior"].fillna(train_df[TARGET_COL].mean()))
        rows.append({
            "val_season": season, "n_train_seasons": train_df["season"].nunique(),
            "n_train_rows": len(train_df), "n_val_rows": len(val_df),
            "mae_model": mae, "mae_naive": naive,
        })
        print(f"  val={season}: trained on {train_df['season'].nunique()} earlier season(s) "
              f"({len(train_df)} rows) -> MAE model={mae:.3f} naive={naive:.3f}")
    return pd.DataFrame(rows)


def run_walk_forward() -> pd.DataFrame:
    df = load_features()
    rows = []

    for fold_start in FOLD_STARTS:
        fold_end = fold_start + FOLD_WIDTH - 1
        train_mask = (df["season"] != BACKTEST_SEASON) | (
            (df["season"] == BACKTEST_SEASON) & (df["GW"] < fold_start)
        )
        test_mask = (
            (df["season"] == BACKTEST_SEASON)
            & (df["GW"] >= fold_start)
            & (df["GW"] <= fold_end)
        )
        train_df, test_df = df[train_mask], df[test_mask]
        if test_df.empty or train_df.empty:
            continue

        train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df[TARGET_COL], categorical_feature=CAT_COLS)
        model = lgb.train(PARAMS, train_set, num_boost_round=NUM_ROUNDS)

        preds = model.predict(test_df[FEATURE_COLS])
        test_df = test_df.assign(pred=preds)

        for gw, gw_df in test_df.groupby("GW"):
            mae_model = mean_absolute_error(gw_df[TARGET_COL], gw_df["pred"])
            mae_naive = mean_absolute_error(
                gw_df[TARGET_COL], gw_df["season_pts_mean_prior"].fillna(train_df[TARGET_COL].mean())
            )
            rows.append({
                "GW": gw, "fold_trained_before_gw": fold_start,
                "n_players": len(gw_df), "mae_model": mae_model, "mae_naive": mae_naive,
            })
        print(f"fold trained on data before GW{fold_start}: predicted GW{fold_start}-{fold_end} "
              f"({len(test_df)} rows) -> mean MAE {test_df.assign(err=(test_df[TARGET_COL]-preds).abs())['err'].mean():.3f}")

    return pd.DataFrame(rows).sort_values("GW")


def cross_season_holdout_check(latest_n_seasons: list[str] | None = None) -> dict:
    """(a) above, restated compactly for the record: train on all seasons
    except the most recent complete one, evaluate on that held-out season."""
    df = load_features()
    train_df = df[df["season"] != BACKTEST_SEASON]
    val_df = df[df["season"] == BACKTEST_SEASON]
    train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df[TARGET_COL], categorical_feature=CAT_COLS)
    model = lgb.train(PARAMS, train_set, num_boost_round=NUM_ROUNDS)
    preds = model.predict(val_df[FEATURE_COLS])
    mae = mean_absolute_error(val_df[TARGET_COL], preds)
    naive = mean_absolute_error(val_df[TARGET_COL], val_df["season_pts_mean_prior"].fillna(train_df[TARGET_COL].mean()))
    return {"season_holdout_mae_model": mae, "season_holdout_mae_naive": naive, "n_train": len(train_df), "n_val": len(val_df)}


if __name__ == "__main__":
    print(f"Loading features and seasons available: ", end="")
    df = load_features()
    print(sorted(df["season"].unique()))

    print("\n=== (a) Cross-season holdout: train on all other seasons, predict all of "
          f"{BACKTEST_SEASON} ===")
    holdout = cross_season_holdout_check()
    print(holdout)

    print(f"\n=== (b) Within-season walk-forward: expanding-window retrain through "
          f"{BACKTEST_SEASON}, predicting blocks of {FOLD_WIDTH} GWs ahead each time ===")
    results = run_walk_forward()
    results.to_csv(DATA_DIR / "backtest_results.csv", index=False)
    print(f"\nSaved -> data/backtest_results.csv ({len(results)} gameweek rows)")

    print("\nSummary by fold (does accuracy improve as the season -- and thus this "
          "season's own accumulated data -- progresses?):")
    print(results.groupby("fold_trained_before_gw")[["mae_model", "mae_naive"]].mean())

    overall_model = (results["mae_model"] * results["n_players"]).sum() / results["n_players"].sum()
    overall_naive = (results["mae_naive"] * results["n_players"]).sum() / results["n_players"].sum()
    print(f"\nOverall walk-forward MAE: model={overall_model:.3f} naive={overall_naive:.3f}")
    print(f"(for comparison, single train/holdout-season split MAE: model="
          f"{holdout['season_holdout_mae_model']:.3f} naive={holdout['season_holdout_mae_naive']:.3f})")

    print(f"\n=== (c) Rolling-origin cross-season CV: is MAE~{holdout['season_holdout_mae_model']:.2f} "
          f"stable across seasons, or one lucky draw? ===")
    rolling = rolling_origin_cv()
    if not rolling.empty:
        rolling.to_csv(DATA_DIR / "rolling_origin_results.csv", index=False)
        print(f"\nSpread across {len(rolling)} validation seasons: "
              f"mean={rolling['mae_model'].mean():.3f} std={rolling['mae_model'].std():.3f} "
              f"min={rolling['mae_model'].min():.3f} max={rolling['mae_model'].max():.3f}")
