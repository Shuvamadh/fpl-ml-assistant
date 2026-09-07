"""Head-to-head: single-stage regressor vs component model vs a blend, on the
held-out 2025-26 season, judged on the metrics that actually matter for squad
decisions (not just raw MAE, which is dominated by non-playing rows).

Run after ``python src/components.py`` (which saves the component models):

    python src/eval_models.py
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

import components as C
from features import FEATURE_COLS, TARGET_COL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VAL_SEASON = "2025-26"
SINGLE_MIN_SEASON = "2020-21"


def _single_stage_holdout(val_season=VAL_SEASON) -> pd.DataFrame:
    """Reproduce train_model.py's regressor on the same split and return its
    2025-26 predictions keyed by (season, GW, element)."""
    df = pd.read_parquet(DATA_DIR / "features.parquet",
                         columns=list(dict.fromkeys(
                             FEATURE_COLS + [TARGET_COL, "season", "GW", "element",
                                             "minutes", "season_gp_prior"])))
    df = df[(df["season_gp_prior"] >= 1) & (df["season"] >= SINGLE_MIN_SEASON)].copy()
    df["position"] = df["position"].astype("category")
    tr = df[df["season"] < val_season]
    va = df[df["season"] == val_season].copy()
    dtrain = lgb.Dataset(tr[FEATURE_COLS], label=tr[TARGET_COL], categorical_feature=["position"])
    params = {"objective": "regression", "metric": "mae", "learning_rate": 0.03,
              "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=1200)
    va["pred_single"] = model.predict(va[FEATURE_COLS])
    return va[["season", "GW", "element", "minutes", TARGET_COL, "pred_single", "season_gp_prior"]]


def _component_holdout(val_season=VAL_SEASON) -> pd.DataFrame:
    full_raw = C._load_raw()
    raw = full_raw[full_raw["season"] >= C.MIN_SEASON].copy()
    feat, _ = C.assemble(raw, team_df=full_raw)
    va = feat[feat["season"] == val_season].copy()
    va = C.predict_frame(va, C.load_models())
    return va[["season", "GW", "element", "minutes", "total_points",
               "pred_points", "pred_points_if_starts", "start_probability"]].rename(
        columns={"pred_points": "pred_comp"})


def _decision_metrics(g: pd.DataFrame, col: str) -> dict:
    """g: one gameweek. col: prediction column."""
    out = {}
    rho, _ = spearmanr(g[TARGET_COL], g[col])
    out["rho"] = rho
    cap_model = g.loc[g[col].idxmax(), TARGET_COL]
    out["captain_regret"] = g[TARGET_COL].max() - cap_model
    out["captain_lift"] = cap_model - g[TARGET_COL].mean()
    # top-11 by model vs top-11 by season PPG, actual points
    top_model = g.nlargest(11, col)[TARGET_COL].sum()
    top_ppg = g.nlargest(11, "season_pts_mean_prior")[TARGET_COL].sum() \
        if "season_pts_mean_prior" in g else np.nan
    out["top11_gain_vs_ppg"] = top_model - top_ppg
    return out


def _bucket(m):
    return np.where(m == 0, "0 (DNP)", np.where(m < 60, "1-59", "60+"))


def main():
    print("training single-stage holdout ...")
    s = _single_stage_holdout()
    print("scoring component holdout ...")
    c = _component_holdout()

    m = s.merge(c[["season", "GW", "element", "pred_comp", "pred_points_if_starts",
                   "start_probability"]],
                on=["season", "GW", "element"], how="inner")
    print(f"matched rows: {len(m)}")

    # --- blend weight chosen on the FIRST HALF of the season, evaluated on the
    #     second, so the weight isn't fit and judged on the same rows ---
    mid = m["GW"].median()
    fit_part = m[m["GW"] <= mid]
    best_w, best_mae = 0.5, 1e9
    for w in np.linspace(0, 1, 21):
        pred = w * fit_part["pred_comp"] + (1 - w) * fit_part["pred_single"]
        mae = mean_absolute_error(fit_part[TARGET_COL], pred)
        if mae < best_mae:
            best_mae, best_w = mae, w
    m["pred_blend"] = best_w * m["pred_comp"] + (1 - best_w) * m["pred_single"]
    test_part = m[m["GW"] > mid]
    print(f"blend weight on component (fit on GW<= {mid:.0f}): {best_w:.2f}")

    preds = {"single": "pred_single", "component": "pred_comp", "blend": "pred_blend"}

    print("\n=== overall MAE (all rows) ===")
    for name, col in preds.items():
        print(f"  {name:10} {mean_absolute_error(m[TARGET_COL], m[col]):.4f}")

    print("\n=== MAE by minutes bucket ===")
    m["mb"] = _bucket(m["minutes"])
    for b, gb in m.groupby("mb"):
        row = "  ".join(f"{name}={mean_absolute_error(gb[TARGET_COL], gb[col]):.3f}"
                        for name, col in preds.items())
        print(f"  {b:8} n={len(gb):>6}  {row}")

    print("\n=== decision metrics (per GW, mean over the season) ===")
    for name, col in preds.items():
        rows = [_decision_metrics(g, col) for _, g in m.groupby("GW") if len(g) >= 30]
        d = pd.DataFrame(rows).mean()
        print(f"  {name:10} rho={d['rho']:.3f}  captain_regret={d['captain_regret']:.2f}  "
              f"captain_lift={d['captain_lift']:.2f}  top11_gain_vs_ppg={d['top11_gain_vs_ppg']:+.2f}")

    # second-half only, to check the blend weight generalised
    print(f"\n=== 2nd-half-only MAE (GW > {mid:.0f}, blend weight held out) ===")
    for name, col in preds.items():
        print(f"  {name:10} {mean_absolute_error(test_part[TARGET_COL], test_part[col]):.4f}")


if __name__ == "__main__":
    main()
