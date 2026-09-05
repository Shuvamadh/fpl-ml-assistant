"""Train a LightGBM model to predict a player's FPL points in a gameweek,
from pre-gameweek form/fixture features. Season-based split (train on older
seasons, validate on the most recent) so validation mimics real deployment.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from features import FEATURE_COLS, TARGET_COL

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

VAL_SEASON = "2025-26"
# seasons before this lack position/team/xG tracking in the archive (mostly-NaN
# rows for those columns) -- excluding them cuts memory pressure and training
# noise for little modeling loss; see README for the full tradeoff discussion.
MIN_SEASON = "2020-21"
CAT_COLS = ["position"]


def load_features() -> pd.DataFrame:
    # "minutes"/"GW"/"element" aren't model features (minutes is post-hoc only
    # -- using this gameweek's actual minutes as an input would be leakage);
    # they're loaded purely to segment/evaluate predictions honestly below.
    needed = list(dict.fromkeys(
        FEATURE_COLS + [TARGET_COL, "season", "season_gp_prior", "minutes", "GW", "element"]
    ))
    # parquet (columnar, typed, float32) is both smaller on disk and far
    # cheaper to read a column subset from than the old CSV -- reads only
    # `needed` columns without pandas' C-parser tokenizing the whole file
    # first, which is what was spiking memory before.
    df = pd.read_parquet("data/features.parquet", columns=needed)
    df = df[(df["season_gp_prior"] >= 1) & (df["season"] >= MIN_SEASON)].copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def train():
    # built and torn down one piece at a time -- on a memory-constrained
    # machine, holding df + train_df + val_df + both Datasets alive at once
    # was enough to push LightGBM's internal allocations into OOM; `del`ing
    # each source frame right after LightGBM has binarized it into a Dataset
    # keeps peak usage well below the naive "keep everything in scope" version.
    df = load_features()
    train_df = df[df["season"] != VAL_SEASON].copy()
    val_df = df[df["season"] == VAL_SEASON].copy()
    del df

    # hard guard: the moment a newer season's rows land in history_gws.csv
    # (data/raw/gws_2026-27.csv already exists as of this writing), this
    # stops the model from silently training on the future and validating
    # on the past if VAL_SEASON isn't bumped to match.
    assert (train_df["season"] <= VAL_SEASON).all(), (
        f"train_df contains seasons newer than VAL_SEASON={VAL_SEASON} -- "
        f"bump VAL_SEASON before retraining, or you're training on the future."
    )

    train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df[TARGET_COL], categorical_feature=CAT_COLS)
    del train_df
    val_set = lgb.Dataset(val_df[FEATURE_COLS], label=val_df[TARGET_COL], categorical_feature=CAT_COLS, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    val_pred = model.predict(val_df[FEATURE_COLS], num_iteration=model.best_iteration)
    val_df = val_df.assign(pred=val_pred)
    mae = mean_absolute_error(val_df[TARGET_COL], val_pred)
    naive_mae = mean_absolute_error(
        val_df[TARGET_COL], val_df["season_pts_mean_prior"].fillna(val_df[TARGET_COL].mean())
    )
    print(f"\nValidation MAE (model):        {mae:.3f}")
    print(f"Validation MAE (naive-avg baseline): {naive_mae:.3f}")
    print("(raw MAE above is dominated by non-players: most rows score 0 and the "
          "model correctly predicts near-0 for them -- see the segmented/decision "
          "metrics below for what this actually means for squad decisions)")

    evaluate_decisions(val_df)

    imp = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURE_COLS)
    print("\nTop features by gain:")
    print(imp.sort_values(ascending=False).head(15))

    model.save_model(str(MODEL_DIR / "points_model.txt"))
    print(f"\nSaved model -> {MODEL_DIR / 'points_model.txt'}")
    return model


def evaluate_decisions(val_df: pd.DataFrame):
    """Metrics that reflect what the app actually does with these predictions
    -- ranking players and picking captains/lineups -- rather than raw MAE,
    which is dominated by the ~57-59% of rows that are non-playing (0
    minutes, 0 points) and trivially easy to get right."""
    def bucket(m):
        if m == 0:
            return "0 min (didn't play)"
        if m < 60:
            return "1-59 min (sub/partial)"
        return "60+ min (started)"

    val_df = val_df.assign(minutes_bucket=val_df["minutes"].apply(bucket))
    print("\n--- MAE by actual minutes played (the honest breakdown) ---")
    for bucket_name, g in val_df.groupby("minutes_bucket"):
        print(f"  {bucket_name:<24} n={len(g):>6}  MAE={mean_absolute_error(g[TARGET_COL], g['pred']):.3f}")

    rhos = []
    captain_regret, captain_lift = [], []
    unconstrained_gain = []
    for gw, g in val_df.groupby("GW"):
        if len(g) < 10:
            continue
        rho, _ = spearmanr(g[TARGET_COL], g["pred"])
        if not np.isnan(rho):
            rhos.append(rho)

        model_captain_actual = g.loc[g["pred"].idxmax(), TARGET_COL]
        perfect_captain_actual = g[TARGET_COL].max()
        pool_mean_actual = g[TARGET_COL].mean()
        captain_regret.append(perfect_captain_actual - model_captain_actual)
        captain_lift.append(model_captain_actual - pool_mean_actual)

        # unconstrained top-11 by model vs top-11 by season-so-far PPG, no
        # formation/budget constraints -- a directional check on whether the
        # model's ranking beats the simplest sensible baseline a manager
        # could use without any tool at all.
        top11_model = g.nlargest(11, "pred")[TARGET_COL].sum()
        top11_ppg = g.nlargest(11, "season_pts_mean_prior")[TARGET_COL].sum()
        unconstrained_gain.append(top11_model - top11_ppg)

    print(f"\n--- Decision-quality metrics (val season, per gameweek, averaged) ---")
    print(f"  Within-GW Spearman rho (pred vs actual, ranking quality): {np.mean(rhos):.3f}")
    print(f"  Captain regret (perfect captain - model's captain, actual pts): {np.mean(captain_regret):.2f}")
    print(f"  Captain lift over pool average: {np.mean(captain_lift):.2f}")
    print(f"  Unconstrained top-11-by-model vs top-11-by-PPG, actual pts gained: "
          f"{np.mean(unconstrained_gain):+.2f} per GW")


def train_two_stage():
    """Hurdle model: P(minutes>=60) classifier x E[points | minutes>=60]
    regressor, combined as p*E[points|started] + (1-p)*not_started_baseline.

    Rationale (from a peer review): "minutes/starts dominate feature
    importance" is the single-stage model telling us it's mostly a
    will-he-play classifier wearing a regressor's clothes. Splitting the two
    questions apart should make both halves easier, AND the classifier's
    P(started) output is directly useful as an explicit rotation-risk column
    in the GUI -- which single-stage pred_points doesn't expose at all.

    Evaluated against train_model.py's `train()` on the SAME held-out season,
    with the peer's explicit caveat applied: the regressor's own accuracy is
    compared against a naive baseline computed on the SAME minutes>=60-only
    rows, not the all-rows naive baseline -- comparing against the wrong
    (easier) baseline would make the improvement look larger than it is.
    """
    df = load_features()
    train_df = df[df["season"] != VAL_SEASON].copy()
    val_df = df[df["season"] == VAL_SEASON].copy()
    del df
    assert (train_df["season"] <= VAL_SEASON).all(), "training on a season newer than VAL_SEASON"

    y_started_train = (train_df["minutes"] >= 60).astype(int)
    y_started_val = (val_df["minutes"] >= 60).astype(int)

    clf_set = lgb.Dataset(train_df[FEATURE_COLS], label=y_started_train, categorical_feature=CAT_COLS)
    clf_val_set = lgb.Dataset(val_df[FEATURE_COLS], label=y_started_val, categorical_feature=CAT_COLS, reference=clf_set)
    clf_params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
    }
    clf = lgb.train(
        clf_params, clf_set, num_boost_round=1000,
        valid_sets=[clf_val_set], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    print(f"Minutes>=60 classifier: best AUC = {clf.best_score['valid_0']['auc']:.3f} "
          f"(best_iteration={clf.best_iteration})")

    started_train = train_df[train_df["minutes"] >= 60]
    started_val = val_df[val_df["minutes"] >= 60]
    not_started_baseline = train_df.loc[train_df["minutes"] < 60, TARGET_COL].mean()

    reg_set = lgb.Dataset(started_train[FEATURE_COLS], label=started_train[TARGET_COL], categorical_feature=CAT_COLS)
    reg_val_set = lgb.Dataset(started_val[FEATURE_COLS], label=started_val[TARGET_COL], categorical_feature=CAT_COLS, reference=reg_set)
    reg_params = {
        "objective": "regression", "metric": "mae", "learning_rate": 0.03,
        "num_leaves": 31, "min_data_in_leaf": 30, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
    }
    reg = lgb.train(
        reg_params, reg_set, num_boost_round=2000,
        valid_sets=[reg_val_set], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    del train_df

    # regressor's own accuracy, on ONLY the rows it was fit for -- compared
    # against a naive baseline computed on those SAME rows (the peer's caveat:
    # comparing against the all-rows baseline would be an easier, misleading
    # target since minutes>=60 rows have higher variance than the population).
    reg_pred_started = reg.predict(started_val[FEATURE_COLS], num_iteration=reg.best_iteration)
    reg_mae_started = mean_absolute_error(started_val[TARGET_COL], reg_pred_started)
    naive_mae_started = mean_absolute_error(
        started_val[TARGET_COL], started_val["season_pts_mean_prior"].fillna(started_train[TARGET_COL].mean())
    )
    print(f"\nRegressor-only, on minutes>=60 rows (n={len(started_val)}): "
          f"MAE={reg_mae_started:.3f} vs naive-on-same-rows={naive_mae_started:.3f}")

    # combined hurdle prediction on the FULL val set, for fair comparison
    # against the single-stage model's overall MAE.
    p_started = clf.predict(val_df[FEATURE_COLS], num_iteration=clf.best_iteration)
    reg_pred_all = reg.predict(val_df[FEATURE_COLS], num_iteration=reg.best_iteration)
    combined = p_started * reg_pred_all + (1 - p_started) * not_started_baseline
    combined_mae = mean_absolute_error(val_df[TARGET_COL], combined)
    print(f"\nTwo-stage combined MAE (full val set): {combined_mae:.3f}")
    print("(compare to single-stage train_model.train()'s MAE printed separately -- "
          "run both and diff by hand, they use the same VAL_SEASON split)")

    clf.save_model(str(MODEL_DIR / "minutes_classifier.txt"))
    reg.save_model(str(MODEL_DIR / "points_regressor_started.txt"))
    import json
    (MODEL_DIR / "two_stage_meta.json").write_text(
        json.dumps({"not_started_baseline": float(not_started_baseline)})
    )
    print(f"\nSaved -> models/minutes_classifier.txt, models/points_regressor_started.txt, "
          f"models/two_stage_meta.json")
    return clf, reg, not_started_baseline


if __name__ == "__main__":
    train()
    print("\n" + "=" * 70)
    print("TWO-STAGE (hurdle) MODEL -- comparison run")
    print("=" * 70)
    train_two_stage()
