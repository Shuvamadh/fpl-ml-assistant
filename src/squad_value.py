"""Reconstruct each squad player's real purchase price and FPL-rule sell price
(not just current market price), from PUBLIC endpoints only:
- entry_transfers(): every transfer made, with cost at time of transfer
- player_summary()'s GW1 history row: price at kickoff for anyone never transferred

FPL's sell-price rule: if a player's price has risen since you bought them,
you only keep HALF the profit (rounded down to the nearest £0.1m); if it's
fallen, you eat the full loss. This matters a lot for realistic transfer
budgeting -- "now_cost" alone overstates your real buying power on risers.
"""
from pathlib import Path

import pandas as pd

from fpl_api import bootstrap_static, entry_transfers, player_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def sell_price(buy_price_m: float, now_price_m: float) -> float:
    if now_price_m <= buy_price_m:
        return now_price_m  # losses are eaten in full
    profit = now_price_m - buy_price_m
    kept_profit = int(profit * 10 // 2) / 10.0  # half profit, floored to 0.1m
    return round(buy_price_m + kept_profit, 1)


def purchase_prices(team_id: int, current_element_ids: list[int]) -> dict[int, float]:
    """element_id -> price paid (£m), for the CURRENTLY held squad only."""
    transfers = entry_transfers(team_id)
    # keep only the most recent buy of each element still in the squad
    buys = {}
    for t in sorted(transfers, key=lambda x: (x["event"], x.get("time", ""))):
        if t["element_in"] in current_element_ids:
            buys[t["element_in"]] = t["element_in_cost"] / 10.0

    missing = [eid for eid in current_element_ids if eid not in buys]
    for eid in missing:
        try:
            hist = player_summary(eid)["history"]
            gw1 = next((h for h in hist if h["round"] == 1), None)
            if gw1:
                buys[eid] = gw1["value"] / 10.0
        except Exception:
            pass
    return buys


def squad_value_report(team_id: int, squad_df: pd.DataFrame) -> pd.DataFrame:
    """squad_df must have columns: element (id), now_cost_m."""
    ids = squad_df["element"].tolist()
    buys = purchase_prices(team_id, ids)
    out = squad_df.copy()
    out["buy_price_m"] = out["element"].map(buys)
    out["buy_price_m"] = out["buy_price_m"].fillna(out["now_cost_m"])
    out["sell_price_m"] = [
        sell_price(b, n) for b, n in zip(out["buy_price_m"], out["now_cost_m"])
    ]
    out["profit_m"] = (out["sell_price_m"] - out["buy_price_m"]).round(1)
    return out


if __name__ == "__main__":
    import sys
    from recommend import load_squad

    team_id = int(sys.argv[1]) if len(sys.argv) > 1 else 8041052
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    squad, meta = load_squad(team_id, event)
    sv = squad_value_report(team_id, squad)
    print(sv[["web_name", "buy_price_m", "now_cost_m", "sell_price_m", "profit_m"]].to_string(index=False))
    print(f"\nTrue sellable squad value: {sv['sell_price_m'].sum():.1f}m (market value: {sv['now_cost_m'].sum():.1f}m)")
    print(f"Total unrealised profit: {sv['profit_m'].sum():.1f}m")
