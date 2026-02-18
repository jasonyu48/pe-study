"""
TuShare PRO permission self-test.

This script helps you verify whether your TuShare account/token has access
to the endpoints we need for survivorship-safe PIT universes and historical P/E.

Usage:
  python data/tushare_selftest.py --token-path data/token
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TuShare permission self-test")
    p.add_argument("--token-path", type=str, default="data/token")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from src.tushare_client import get_pro

    pro = get_pro(args.token_path)

    tests = []

    # (name, callable)
    tests.append(
        (
            "index_weight (CSI300 snapshot)",
            lambda: pro.index_weight(index_code="000300.SH", start_date="20180101", end_date="20180131"),
        )
    )
    tests.append(
        (
            "index_member (CSI300 members)",
            lambda: pro.index_member(index_code="000300.SH"),
        )
    )
    tests.append(
        (
            "stock_basic (A-share master)",
            lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,list_date"),
        )
    )
    tests.append(
        (
            "trade_cal (calendar)",
            lambda: pro.trade_cal(exchange="SSE", start_date="20180101", end_date="20180131"),
        )
    )
    tests.append(
        (
            "daily_basic (P/E TTM example day)",
            lambda: pro.daily_basic(trade_date="20180102", fields="ts_code,trade_date,pe_ttm,close"),
        )
    )

    ok = 0
    for name, fn in tests:
        try:
            df = fn()
            n = 0 if df is None else len(df)
            cols = [] if df is None else list(df.columns)
            print(f"[OK]   {name}: rows={n} cols={cols[:8]}")
            ok += 1
        except Exception as e:
            msg = str(e).replace("\n", " ").strip()
            print(f"[FAIL] {name}: {type(e).__name__}: {msg}")

    print(f"\nSummary: {ok}/{len(tests)} endpoints accessible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

