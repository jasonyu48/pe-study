"""
Fetch China A-share month-start OPEN prices using TuShare PRO `daily` (by trade_date).

Why:
  - For an execution model "signal at month-end close -> trade at next month open",
    the natural forward return is open-to-open using month-start opens.
  - A-share stocks can be suspended on the first trading day of a month. If we only
    fetch the market-wide first trading day, many names will have missing opens.
  - This script fetches the first N trading days of each month and, for each stock,
    takes the first available open within that window (per-stock "month-start open").

Output (long format):
  columns: month, date, symbol, open_raw, adj_factor, open
  where:
    - month is YYYY-MM
    - date is the first trading day (within the window) that the stock traded (datetime64[ns])
    - open_raw is that day's raw open price
    - open is qfq-equivalent open (open_raw * adj_factor; normalized constant cancels in returns)

Example:
  python data/fetch_cn_prices_monthstart_open_tushare.py --start 2016-01-01 --end 2025-12-31 --out data/cn_prices_monthstart_open.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tushare_client import get_pro  # noqa: E402


def _fetch_adj_factor_all(pro, trade_date: str) -> pd.DataFrame:
    af = pro.adj_factor(trade_date=trade_date, fields="ts_code,trade_date,adj_factor")
    if af is None or af.empty:
        return pd.DataFrame(columns=["symbol", "adj_factor"])
    af = af.rename(columns={"ts_code": "symbol"})
    af["symbol"] = af["symbol"].astype(str).str.strip()
    af["adj_factor"] = pd.to_numeric(af["adj_factor"], errors="coerce")
    return af[["symbol", "adj_factor"]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch month-start open prices using TuShare daily(trade_date=...).")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=str, default="data/cn_prices_monthstart_open.parquet")
    p.add_argument("--token-path", type=str, default="data/token")
    p.add_argument(
        "--window-trading-days",
        type=int,
        default=10,
        help="Fetch first N trading days per month and take first available open per stock.",
    )
    p.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between API calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pro = get_pro(args.token_path)
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    if cal is None or cal.empty:
        raise SystemExit("TuShare returned empty trade_cal; check token/date range.")
    cal = cal[cal["is_open"] == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    cal["month"] = cal["cal_date"].dt.to_period("M").astype(str)
    # first N trading days per month
    cal = cal.sort_values(["month", "cal_date"])
    win_n = int(args.window_trading_days)
    if win_n <= 0:
        raise SystemExit("--window-trading-days must be positive.")
    month_dates = cal.groupby("month")["cal_date"].apply(lambda s: list(s.iloc[:win_n])).to_dict()

    frames: list[pd.DataFrame] = []
    for month in tqdm(sorted(month_dates.keys()), total=len(month_dates), desc="Fetching month-start opens", unit="month"):
        dates = month_dates.get(month, [])
        if not dates:
            continue

        assigned: set[str] = set()
        month_rows: list[pd.DataFrame] = []

        for trade_date in dates:
            td = pd.Timestamp(trade_date).strftime("%Y%m%d")
            df = pro.daily(trade_date=td, fields="ts_code,trade_date,open")
            if df is None or df.empty:
                time.sleep(float(args.sleep))
                continue
            af = _fetch_adj_factor_all(pro, td)

            df = df.rename(columns={"ts_code": "symbol"})
            df["date"] = pd.to_datetime(df["trade_date"])
            df["month"] = month
            df["symbol"] = df["symbol"].astype(str).str.strip()
            df["open_raw"] = pd.to_numeric(df["open"], errors="coerce")
            df = df.merge(af, on="symbol", how="left")
            df["open"] = df["open_raw"] * df["adj_factor"]

            df = df.dropna(subset=["symbol", "date", "open"])
            if assigned:
                df = df[~df["symbol"].isin(list(assigned))]
            if not df.empty:
                month_rows.append(df[["month", "date", "symbol", "open_raw", "adj_factor", "open"]])
                assigned.update(df["symbol"].astype(str).tolist())

            time.sleep(float(args.sleep))

        if month_rows:
            mdf = pd.concat(month_rows, ignore_index=True)
            # ensure one row per (month, symbol): first tradable day within the window
            mdf = mdf.sort_values(["month", "symbol", "date"]).groupby(["month", "symbol"], as_index=False).first()
            frames.append(mdf)

    if not frames:
        raise SystemExit("No frames fetched; check TuShare permissions.")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["month", "date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} rows={len(out):,} months={out['month'].nunique()} symbols={out['symbol'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

