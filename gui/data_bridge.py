"""Bridge between the GUI and the existing src/ pipeline. Runs the (slow,
network-bound) refresh on a background thread so the UI never freezes."""
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _manager_gw_history(standings, workers: int = 12) -> pd.DataFrame:
    """GW-by-GW history (points, bench points, value, overall FPL rank, chip)
    for every manager in the league -- built purely from src.fpl_api.entry_history,
    which mini_league.py already uses for league_rank_progression. One row per
    (manager, gameweek)."""
    import concurrent.futures as cf
    from fpl_api import entry_history

    def _fetch(row):
        try:
            hist = entry_history(row["entry"])["current"]
            for h in hist:
                h = dict(h)
            return row["entry_name"], hist
        except Exception:
            return row["entry_name"], []

    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for entry_name, hist in ex.map(_fetch, [r for _, r in standings.iterrows()]):
            for h in hist:
                rows.append({**h, "entry_name": entry_name})
    return pd.DataFrame(rows)


def build_fun_stats(league_squads, insights, manager_hist, my_entry_id: int) -> dict:
    """Derives "fun stats for arguing with friends" purely from data already
    pulled elsewhere in the refresh (league_squads, league_insights,
    manager_hist) -- no new API surface, no src/ changes."""
    out = {}

    # Bench points left to rot, season-to-date, per manager.
    if manager_hist is not None and not manager_hist.empty and "points_on_bench" in manager_hist:
        bench = (
            manager_hist.groupby("entry_name")["points_on_bench"].sum()
            .rename("bench_points").reset_index()
            .sort_values("bench_points", ascending=False)
        )
        out["bench_points"] = bench
    else:
        out["bench_points"] = pd.DataFrame(columns=["entry_name", "bench_points"])

    # Squad value growth (start of season -> now), per manager. `value` is in
    # tenths of a million, FPL-API-wide.
    if manager_hist is not None and not manager_hist.empty and "value" in manager_hist:
        mh = manager_hist.sort_values("event")
        first = mh.groupby("entry_name")["value"].first()
        last = mh.groupby("entry_name")["value"].last()
        growth = ((last - first) / 10.0).rename("value_growth_m").reset_index()
        growth = growth.sort_values("value_growth_m", ascending=False)
        out["value_growth"] = growth
    else:
        out["value_growth"] = pd.DataFrame(columns=["entry_name", "value_growth_m"])

    # Biggest single-GW swing in overall FPL rank (not league rank -- FPL
    # doesn't expose historical mini-league rank, only overall rank per GW).
    if manager_hist is not None and not manager_hist.empty and "overall_rank" in manager_hist:
        mh = manager_hist.sort_values("event").copy()
        mh["rank_swing"] = mh.groupby("entry_name")["overall_rank"].diff().abs()
        swings = (
            mh.groupby("entry_name")["rank_swing"].max()
            .rename("rank_swing").reset_index()
            .dropna().sort_values("rank_swing", ascending=False)
        )
        out["rank_swings"] = swings
    else:
        out["rank_swings"] = pd.DataFrame(columns=["entry_name", "rank_swing"])

    # Template adherence + differentials-owned count, per manager -- from this
    # gameweek's squads plus league-wide ownership already computed in insights.
    if league_squads is not None and not league_squads.empty and insights is not None:
        ownership = insights["ownership"]
        template_ids = set(ownership[ownership["owned_pct"] >= 50]["element"])
        diff_threshold = max(1, insights["n_managers"] // 5)
        diff_ids = set(ownership[ownership["owned_by_n"] <= diff_threshold]["element"])

        squad_sizes = league_squads.groupby("entry_name")["element"].nunique()
        template_counts = (
            league_squads[league_squads["element"].isin(template_ids)]
            .groupby("entry_name")["element"].nunique()
        )
        diff_counts = (
            league_squads[league_squads["element"].isin(diff_ids)]
            .groupby("entry_name")["element"].nunique()
        )
        adherence = pd.DataFrame({
            "entry_name": squad_sizes.index,
            "squad_size": squad_sizes.values,
        })
        adherence["template_count"] = adherence["entry_name"].map(template_counts).fillna(0).astype(int)
        adherence["template_pct"] = (adherence["template_count"] / adherence["squad_size"] * 100).round(0)
        adherence["differentials_owned"] = adherence["entry_name"].map(diff_counts).fillna(0).astype(int)
        adherence = adherence.sort_values("template_pct", ascending=False)
        out["template_adherence"] = adherence
    else:
        out["template_adherence"] = pd.DataFrame(
            columns=["entry_name", "squad_size", "template_count", "template_pct", "differentials_owned"]
        )

    return out


class RefreshWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, team_id: int, event: int, league_id: int | None = None):
        super().__init__()
        self.team_id = team_id
        self.event_id = event
        self.league_id = league_id

    def run(self):
        try:
            self.progress.emit("Scoring players (pulling live per-player data)...")
            import predict
            predictions = predict.score_players()
            predictions.to_csv(predict.DATA_DIR / "player_predictions.csv", index=False)

            self.progress.emit("Loading your squad...")
            import recommend
            squad, meta = recommend.load_squad(self.team_id, self.event_id)

            self.progress.emit("Reconstructing squad value (buy/sell prices)...")
            import squad_value
            sv = squad_value.squad_value_report(self.team_id, squad)
            squad = squad.merge(
                sv[["element", "buy_price_m", "sell_price_m", "profit_m"]],
                on="element", how="left",
            )

            self.progress.emit("Optimising starting XI...")
            xi, bench = recommend.best_starting_xi(squad)

            self.progress.emit("Scanning for transfer upgrades...")
            transfers = recommend.suggest_transfers(squad, meta["bank"])

            result = {
                "predictions": predictions,
                "squad": squad,
                "meta": meta,
                "xi": xi,
                "bench": bench,
                "transfers": transfers,
                "standings": None,
                "insights": None,
                "league_name": None,
                "fixture_matrix": None,
                "rank_progression": None,
                "league_squads": None,
                "manager_hist": None,
                "eff_ownership": None,
                "league_form": None,
                "captaincy_impact": None,
                "fun_stats": None,
                "projection": None,
                "rival_transfers": None,
                "chips": None,
            }

            # --- chip strategy (squad-only, doesn't need a league) ---
            self.progress.emit("Optimising wildcard / free-hit squads...")
            try:
                import chips
                blank_double = chips.detect_blank_double_gameweeks(n=5)
                result["chips"] = {
                    "blank_double": blank_double,
                    "wildcard": chips.wildcard_squad(self.team_id, squad),
                    "free_hit": chips.free_hit_squad(self.team_id, squad),
                    "triple_captain": chips.triple_captain_candidates(squad, blank_double),
                    "bench_boost": chips.bench_boost_value(
                        squad, bench["element"].tolist() if "element" in bench else []
                    ),
                    "used": chips.chips_used(self.team_id),
                }
            except Exception as e:
                # the ILP needs pulp; degrade to a message rather than killing
                # the whole refresh
                result["chips"] = {"error": str(e)}

            self.progress.emit("Building fixture difficulty matrix...")
            try:
                import fixtures_fdr
                result["fixture_matrix"] = fixtures_fdr.fixture_difficulty_matrix(n=5)
            except Exception:
                result["fixture_matrix"] = None

            if self.league_id:
                self.progress.emit("Pulling mini-league standings...")
                import mini_league
                standings = mini_league.league_standings(self.league_id)
                self.progress.emit(f"Pulling squads for {len(standings)} rival managers...")
                league_squads = mini_league.build_league_squads(standings, self.event_id)
                insights = mini_league.league_insights(league_squads, self.team_id)
                result["standings"] = standings
                result["insights"] = insights
                result["league_name"] = standings.attrs.get("league_name", "Mini League")

                result["league_squads"] = league_squads

                self.progress.emit(f"Pulling GW-by-GW rank progression for {len(standings)} managers...")
                try:
                    result["rank_progression"] = mini_league.league_rank_progression(standings)
                except Exception:
                    result["rank_progression"] = None

                self.progress.emit("Projecting this gameweek's league winner...")
                try:
                    import league_projection
                    result["projection"] = league_projection.project_gw_winner(
                        league_squads, standings, self.event_id
                    )
                    result["rival_transfers"] = league_projection.per_manager_transfer_suggestions(
                        league_squads
                    )
                except Exception:
                    result["projection"] = None
                    result["rival_transfers"] = None

                self.progress.emit("Crunching league fun stats...")
                try:
                    manager_hist = _manager_gw_history(standings)
                    result["manager_hist"] = manager_hist
                    result["fun_stats"] = build_fun_stats(
                        league_squads, insights, manager_hist, self.team_id
                    )
                except Exception:
                    result["fun_stats"] = None

                self.progress.emit("Computing effective ownership and league form...")
                try:
                    import league_extras
                    result["eff_ownership"] = league_extras.effective_ownership(league_squads)
                    result["league_form"] = league_extras.league_form(
                        result.get("manager_hist"), last_n=4
                    )
                    result["captaincy_impact"] = league_extras.captaincy_impact(
                        league_squads, predictions, self.team_id
                    )
                except Exception:
                    result["eff_ownership"] = None
                    result["league_form"] = None
                    result["captaincy_impact"] = None

            self.progress.emit("Done.")
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


def run_refresh(team_id: int, event: int, on_progress, on_done, on_error, league_id: int | None = None):
    thread = QThread()
    worker = RefreshWorker(team_id, event, league_id)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.progress.connect(on_progress)
    worker.finished.connect(on_done)
    worker.failed.connect(on_error)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
