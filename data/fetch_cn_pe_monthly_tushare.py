"""
Fetch monthly P/E (TTM) snapshots from TuShare PRO `daily_basic`.

Output long-format factor file:
  columns: date, symbol, pe
where date is the month-end trading day (datetime64[ns]) and pe is pe_ttm.

Typical workflow:
  1) Build PIT membership (e.g., CSI300) with `month,symbol`
  2) Fetch P/E snapshots for the month-end trade dates
  3) Join with PIT membership at backtest time

Example:
  python data/fetch_cn_pe_monthly_tushare.py --start 2010-01-01 --end 2025-12-31 --out data/cn_pe_monthly.parquet
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch monthly P/E snapshots (pe_ttm) from TuShare daily_basic.")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=str, default="data/cn_pe_monthly.parquet")
    p.add_argument("--token-path", type=str, default="data/token")
    p.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between API calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    pro = get_pro(args.token_path)
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    if cal is None or cal.empty:
        raise SystemExit("TuShare returned empty trade_cal; check token/date range.")

    cal = cal[cal["is_open"] == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    cal["month"] = cal["cal_date"].dt.to_period("M").astype(str)
    month_last = cal.groupby("month")["cal_date"].max().sort_values()

    frames: list[pd.DataFrame] = []
    for month, trade_date in tqdm(month_last.items(), total=len(month_last), desc="Fetching P/E", unit="month"):
        td = pd.Timestamp(trade_date).strftime("%Y%m%d")
        df = pro.daily_basic(trade_date=td, fields="ts_code,trade_date,pe_ttm")
        if df is None or df.empty:
            continue
        df = df.rename(columns={"ts_code": "symbol", "pe_ttm": "pe"})
        df["date"] = pd.to_datetime(df["trade_date"])
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
        frames.append(df[["date", "symbol", "pe"]])
        time.sleep(float(args.sleep))

    if not frames:
        raise SystemExit("No P/E frames fetched; check TuShare permissions.")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False) if out_path.suffix.lower() in {".parquet", ".pq"} else out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} rows={len(out):,} months={out['date'].dt.to_period('M').nunique()} symbols={out['symbol'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

