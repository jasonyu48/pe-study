"""
Fetch index daily prices from TuShare and compute monthly market regimes.

Regimes defined per month:
  - up/down: sign of monthly return
  - high/low vol: compare monthly realized vol (std of daily returns within month)
    to the median vol across all months

Default index:
  - attempts to find "中证全指" from index_basic(market="CSI") and use its ts_code

Outputs:
  - monthly regimes parquet/csv: month, mkt_ret, up, vol, high_vol

Example:
  python data/fetch_cn_index_regimes_tushare.py --start 2016-01-01 --end 2025-12-31 --out data/cn_market_regimes.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tushare_client import get_pro  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch index daily and compute monthly regimes.")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=str, default="data/cn_market_regimes.parquet")
    p.add_argument("--token-path", type=str, default="data/token")
    p.add_argument("--ts-code", type=str, default="", help="Index ts_code (optional). If empty, auto-detect 中证全指.")
    p.add_argument("--sleep", type=float, default=0.12)
    return p.parse_args(argv)


def _detect_all_share_ts_code(pro) -> str:
    # Try CSI market index basic; fall back to full search if needed.
    df = pro.index_basic(market="CSI")
    if df is None or df.empty:
        raise SystemExit("TuShare index_basic(market='CSI') returned empty; check permissions.")
    # common names: 中证全指
    cand = df[df["name"].astype(str).str.contains("中证全指", na=False)].copy()
    if cand.empty:
        raise SystemExit("Could not find index named '中证全指' in index_basic(market='CSI').")
    return str(cand.iloc[0]["ts_code"])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pro = get_pro(args.token_path)
    ts_code = args.ts_code.strip() or _detect_all_share_ts_code(pro)

    start = pd.to_datetime(args.start).strftime("%Y%m%d")
    end = pd.to_datetime(args.end).strftime("%Y%m%d")

    # index_daily supports ts_code + date range
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end, fields="ts_code,trade_date,close")
    if df is None or df.empty:
        raise SystemExit(f"TuShare returned empty index_daily for {ts_code}.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    if df.empty:
        raise SystemExit("Index daily data is empty after cleaning.")

    df["ret"] = df["close"].pct_change(fill_method=None)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # Monthly return: last close / first close - 1
    first_close = df.groupby("month")["close"].first()
    last_close = df.groupby("month")["close"].last()
    mkt_ret = (last_close / first_close - 1.0).rename("mkt_ret")

    # Monthly realized vol: std of daily returns within month
    vol = df.groupby("month")["ret"].std(ddof=1).rename("vol")
    out = pd.concat([mkt_ret, vol], axis=1).reset_index().rename(columns={"index": "month"})
    out["up"] = out["mkt_ret"] > 0
    med = float(out["vol"].median(skipna=True)) if out["vol"].notna().any() else float("nan")
    out["high_vol"] = out["vol"] >= med
    out["index_ts_code"] = ts_code

    if out_path.suffix.lower() in {".parquet", ".pq"}:
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} months={len(out):,} index={ts_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

