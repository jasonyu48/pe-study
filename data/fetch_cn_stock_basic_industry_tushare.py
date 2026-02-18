"""
Fetch stock -> industry mapping from TuShare PRO `stock_basic`.

This provides a simple (non-PIT) industry label suitable for first-pass
industry-neutral PE research.

Output:
  columns: symbol, name, industry, market

Example:
  python data/fetch_cn_stock_basic_industry_tushare.py --out data/cn_stock_industry.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tushare_client import get_pro  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch stock_basic industry mapping from TuShare.")
    p.add_argument("--out", type=str, default="data/cn_stock_industry.parquet")
    p.add_argument("--token-path", type=str, default="data/token")
    p.add_argument("--list-status", type=str, default="L", choices=["L", "D", "P"], help="L=listed, D=delisted, P=suspended")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pro = get_pro(args.token_path)
    df = pro.stock_basic(exchange="", list_status=args.list_status, fields="ts_code,name,industry,market")
    if df is None or df.empty:
        raise SystemExit("TuShare returned empty stock_basic; check permissions.")
    df = df.rename(columns={"ts_code": "symbol"}).copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip()

    if out_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} rows={len(df):,} symbols={df['symbol'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

