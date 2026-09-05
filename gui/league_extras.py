"""Extra mini-league analytics, derived from data the refresh already pulls.

Deliberately free of any Qt import so the Streamlit build can use the same
functions instead of reimplementing them. Everything here takes plain
DataFrames and returns plain DataFrames/dicts.

Inputs used throughout:
    league_squads  one row per (manager, player) for a gameweek, with columns
                   entry, entry_name, element, web_name, position, is_captain,
                   is_vice_captain, multiplier, cost_m
    predictions    data/player_predictions.csv (id, pred_points_adj, ...)
"""
import pandas as pd


def effective_ownership(league_squads: pd.DataFrame) -> pd.DataFrame:
    """Effective ownership (EO) per player, within this league.

    EO is the number that actually decides whether a pick helps or hurts you,
    and plain ownership hides it: a player owned by 50% but captained by 40%
    behaves like 90% ownership, because captained copies score double. Anyone
    ranking "most owned" alone is reading the wrong column.

    Returns: web_name, position, owned_n, owned_pct, started_n, captained_n,
             eo_pct, and template_risk.
    """
    if league_squads is None or league_squads.empty:
        return pd.DataFrame(columns=["web_name", "position", "owned_n", "owned_pct",
                                     "started_n", "captained_n", "eo_pct", "template_risk"])

    n_managers = league_squads["entry"].nunique()
    df = league_squads.copy()
    df["is_captain"] = df.get("is_captain", False).fillna(False)
    # multiplier 0 = benched, 1 = starting, 2 = captain, 3 = triple captain
    mult = df.get("multiplier", pd.Series(1, index=df.index)).fillna(0)

    grouped = df.groupby(["element", "web_name", "position"], as_index=False).agg(
        owned_n=("entry", "nunique"),
        captained_n=("is_captain", "sum"),
    )
    started = (
        df[mult >= 1].groupby("element", as_index=False)["entry"].nunique()
        .rename(columns={"entry": "started_n"})
    )
    grouped = grouped.merge(started, on="element", how="left")
    grouped["started_n"] = grouped["started_n"].fillna(0).astype(int)

    grouped["owned_pct"] = (grouped["owned_n"] / n_managers * 100).round(1)
    cap_pct = grouped["captained_n"] / n_managers * 100
    # EO = starting ownership + an extra copy for every captain
    start_pct = grouped["started_n"] / n_managers * 100
    grouped["eo_pct"] = (start_pct + cap_pct).round(1)

    def risk(row):
        if row["eo_pct"] >= 50:
            return "Must-own"
        if row["eo_pct"] >= 20:
            return "Template"
        if row["eo_pct"] > 0:
            return "Differential"
        return "Benched"

    grouped["template_risk"] = grouped.apply(risk, axis=1)
    return grouped.sort_values("eo_pct", ascending=False).reset_index(drop=True)


def head_to_head(league_squads: pd.DataFrame, predictions: pd.DataFrame,
                 entry_a: int, entry_b: int) -> dict:
    """Squad-level comparison of two managers.

    The shared players cancel out -- only the differences can move the gap
    between two managers, so those are what the table shows and what the
    projected swing is computed from.
    """
    a = league_squads[league_squads["entry"] == entry_a]
    b = league_squads[league_squads["entry"] == entry_b]
    if a.empty or b.empty:
        return {"error": "one or both managers not found in this league"}

    preds = predictions[["id", "pred_points_adj"]].rename(columns={"id": "element"})
    a = a.merge(preds, on="element", how="left")
    b = b.merge(preds, on="element", how="left")
    a["pred_points_adj"] = a["pred_points_adj"].fillna(0.0)
    b["pred_points_adj"] = b["pred_points_adj"].fillna(0.0)

    shared_ids = set(a["element"]) & set(b["element"])
    only_a = a[~a["element"].isin(shared_ids)].sort_values("pred_points_adj", ascending=False)
    only_b = b[~b["element"].isin(shared_ids)].sort_values("pred_points_adj", ascending=False)

    def starters_points(df):
        mult = df.get("multiplier", pd.Series(1, index=df.index)).fillna(0)
        return float((df["pred_points_adj"] * mult).sum())

    return {
        "name_a": a["entry_name"].iloc[0],
        "name_b": b["entry_name"].iloc[0],
        "shared_n": len(shared_ids),
        "squad_size": len(a),
        "overlap_pct": round(len(shared_ids) / max(1, len(a)) * 100, 0),
        "only_a": only_a[["web_name", "position", "pred_points_adj"]],
        "only_b": only_b[["web_name", "position", "pred_points_adj"]],
        "unique_pred_a": round(float(only_a["pred_points_adj"].sum()), 2),
        "unique_pred_b": round(float(only_b["pred_points_adj"].sum()), 2),
        "projected_a": round(starters_points(a), 2),
        "projected_b": round(starters_points(b), 2),
        "captain_a": (a[a["is_captain"] == True]["web_name"].iloc[0]
                      if (a["is_captain"] == True).any() else None),
        "captain_b": (b[b["is_captain"] == True]["web_name"].iloc[0]
                      if (b["is_captain"] == True).any() else None),
    }


def player_owners(league_squads: pd.DataFrame, query: str) -> pd.DataFrame:
    """Who in the league owns a given player, and are they starting/captaining.

    Answers the question you actually ask mid-gameweek: "he just scored -- who
    has him?"
    """
    if league_squads is None or league_squads.empty or not query:
        return pd.DataFrame(columns=["web_name", "entry_name", "role"])
    hits = league_squads[
        league_squads["web_name"].str.contains(query, case=False, na=False)
    ].copy()
    if hits.empty:
        return pd.DataFrame(columns=["web_name", "entry_name", "role"])

    def role(r):
        if r.get("is_captain"):
            return "Captain"
        if r.get("is_vice_captain"):
            return "Vice"
        return "Starting" if (r.get("multiplier") or 0) >= 1 else "Bench"

    hits["role"] = hits.apply(role, axis=1)
    order = {"Captain": 0, "Vice": 1, "Starting": 2, "Bench": 3}
    hits["_o"] = hits["role"].map(order)
    return (hits.sort_values(["web_name", "_o", "entry_name"])
            [["web_name", "entry_name", "role"]].reset_index(drop=True))


def league_form(manager_hist: pd.DataFrame, last_n: int = 4) -> pd.DataFrame:
    """Recent form: points over the last N gameweeks per manager, which the
    cumulative standings table completely hides -- someone 40 points behind but
    top of this table is the one actually coming for you.
    """
    if manager_hist is None or manager_hist.empty or "event" not in manager_hist:
        return pd.DataFrame(columns=["entry_name", "recent_points", "gws_counted"])
    mh = manager_hist.sort_values("event")
    max_gw = mh["event"].max()
    window = mh[mh["event"] > max_gw - last_n]
    out = (window.groupby("entry_name", as_index=False)
           .agg(recent_points=("points", "sum"),
                gws_counted=("event", "nunique"),
                bench_wasted=("points_on_bench", "sum")))
    return out.sort_values("recent_points", ascending=False).reset_index(drop=True)


def captaincy_impact(league_squads: pd.DataFrame, predictions: pd.DataFrame,
                     my_entry: int) -> dict:
    """How your captain choice compares with the league's.

    Captaincy is the single biggest source of rank movement, and the useful
    framing is not "is my captain good" but "how many points does my captain
    gain or lose me against what everyone else is doing".
    """
    preds = predictions[["id", "pred_points_adj"]].rename(columns={"id": "element"})
    df = league_squads.merge(preds, on="element", how="left")
    df["pred_points_adj"] = df["pred_points_adj"].fillna(0.0)
    caps = df[df["is_captain"] == True]
    if caps.empty:
        return {"error": "no captain data for this gameweek"}

    mine = caps[caps["entry"] == my_entry]
    my_cap = mine["web_name"].iloc[0] if not mine.empty else None
    my_pred = float(mine["pred_points_adj"].iloc[0]) if not mine.empty else 0.0
    # league-average captain return = what a typical rival gets from theirs
    league_avg = float(caps["pred_points_adj"].mean())

    split = (caps.groupby("web_name", as_index=False)
             .agg(managers=("entry", "nunique"),
                  pred=("pred_points_adj", "first"))
             .sort_values("managers", ascending=False))
    split["pct"] = (split["managers"] / caps["entry"].nunique() * 100).round(0)

    return {
        "my_captain": my_cap,
        "my_captain_pred": round(my_pred, 2),
        "league_avg_captain_pred": round(league_avg, 2),
        # doubled: a captain scores twice, so the edge is on the doubled points
        "edge_vs_league": round((my_pred - league_avg) * 2, 2),
        "most_popular": split.iloc[0]["web_name"] if not split.empty else None,
        "split": split,
    }
