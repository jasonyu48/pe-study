"""
PE factor research:
  - IC vs horizon: 1m, 6m, 12m, 24m forward returns
  - IC vs market regimes: up/down, high/low vol (from index-based monthly regimes)
  - PE variants:
      1) plain: score = -PE
      2) threshold: score = -PE, but if mom <= 0 then assign floor score
      3) industry-neutral: within each industry each month, z-score PE then negate

Inputs (parquet/csv):
  --pe: columns date,symbol,pe  (month-end snapshots)
  --prices: columns date,symbol,close  (month-end closes for all stocks; used for momentum gating)
  --prices-monthstart-open: columns month,date,symbol,open (month-start opens; used for open-to-open forward returns)
  --industry: columns symbol,industry
  --regimes: columns month,mkt_ret,up,vol,high_vol

Output:
  - ic_series.parquet: date, horizon, variant, ic
  - ic_summary.parquet: horizon, variant, regime, mean_ic, std_ic, ic_ir, t_stat, n
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


Variant = Literal["plain", "threshold", "industry_neutral"]


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    raise SystemExit("Unsupported input format. Use .csv or .parquet")


def _write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suf == ".csv":
        df.to_csv(path, index=False)
        return
    raise SystemExit("Unsupported output format. Use .csv or .parquet")


def _to_month_end(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d).dt.to_period("M").dt.to_timestamp("M")


def _prepare_monthly_panels(pe_long: pd.DataFrame, px_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pe = pe_long.copy()
    pe["date"] = pd.to_datetime(pe["date"])
    pe["symbol"] = pe["symbol"].astype(str).str.strip()
    pe["pe"] = pd.to_numeric(pe["pe"], errors="coerce")

    px = px_long.copy()
    px["date"] = pd.to_datetime(px["date"])
    px["symbol"] = px["symbol"].astype(str).str.strip()
    # accept close or adj_close
    price_col = "close" if "close" in px.columns else ("adj_close" if "adj_close" in px.columns else None)
    if price_col is None:
        raise SystemExit("prices file must contain column 'close' or 'adj_close'.")
    px["close"] = pd.to_numeric(px[price_col], errors="coerce")

    pe_w = pe.pivot(index="date", columns="symbol", values="pe").sort_index()
    px_w = px.pivot(index="date", columns="symbol", values="close").sort_index()

    # Align dates and symbols intersection
    common_dates = pe_w.index.intersection(px_w.index)
    common_syms = pe_w.columns.intersection(px_w.columns)
    pe_w = pe_w.reindex(index=common_dates, columns=common_syms)
    px_w = px_w.reindex(index=common_dates, columns=common_syms)
    return pe_w, px_w


def _momentum(px_w: pd.DataFrame, lookback_m: int) -> pd.DataFrame:
    # simple momentum over lookback months
    return px_w / px_w.shift(lookback_m) - 1.0


def _industry_neutral_score(pe_row: pd.Series, industry_map: pd.Series) -> pd.Series:
    """
    Cross-sectional score for one date:
      score = -zscore(PE within industry)
    """
    pe = pd.to_numeric(pe_row, errors="coerce")
    ind = industry_map.reindex(pe.index)
    out = pd.Series(np.nan, index=pe.index, dtype="float64")

    for g, idx in ind.dropna().groupby(ind.dropna()).groups.items():
        vals = pe.loc[list(idx)].astype("float64")
        m = vals.notna() & np.isfinite(vals) & (vals > 0)
        if int(m.sum()) < 3:
            continue
        x = vals[m]
        std = float(x.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            continue
        z = (x - float(x.mean())) / std
        out.loc[x.index] = (-z).astype("float64")
    return out


def _compute_ic_series(
    score: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    *,
    min_names: int = 30,
) -> pd.Series:
    ics = []
    for d in score.index:
        s = score.loc[d]
        r = fwd_ret.loc[d]
        m = s.notna() & r.notna() & np.isfinite(s) & np.isfinite(r)
        if int(m.sum()) < int(min_names):
            ics.append(np.nan)
            continue
        x = s[m].astype(float).values
        y = r[m].astype(float).values
        if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
            ics.append(np.nan)
            continue
        ic, _ = spearmanr(x, y)
        ics.append(float(ic) if np.isfinite(ic) else np.nan)
    return pd.Series(ics, index=score.index, name="ic").astype("float64")


def _summarize_ic(x: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(x, errors="coerce").dropna().astype("float64")
    if s.empty:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "t_stat": float("nan"), "n": 0.0}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    ic_ir = float(mean / std) if std and np.isfinite(std) else float("nan")
    t = float(mean / (std / np.sqrt(len(s)))) if std and np.isfinite(std) else float("nan")
    return {"mean_ic": mean, "std_ic": std, "ic_ir": ic_ir, "t_stat": t, "n": float(len(s))}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PE factor IC research.")
    p.add_argument("--pe", type=str, required=True, help="Month-end PE snapshots: date,symbol,pe")
    p.add_argument("--prices", type=str, required=True, help="Month-end prices: date,symbol,close (all stocks)")
    p.add_argument(
        "--prices-monthstart-open",
        type=str,
        required=True,
        help="Month-start open snapshots: month,date,symbol,open. Forward returns use open-to-open.",
    )
    p.add_argument("--industry", type=str, required=True, help="Stock industry mapping: symbol,industry")
    p.add_argument("--regimes", type=str, required=True, help="Monthly regimes: month,up,high_vol,...")
    p.add_argument("--out-dir", type=str, default="reports/pe_research")
    p.add_argument("--min-names", type=int, default=30, help="Minimum cross-sectional names per month to compute IC")
    p.add_argument("--floor", type=float, default=-1e12, help="Floor score for ineligible names (threshold variant)")
    p.add_argument("--mom-lookback-m", type=int, default=6, help="Momentum lookback in months for threshold PE")
    p.add_argument("--horizons", type=str, default="1,6,12,24", help="Comma-separated forward horizons in months")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pe_long = _read_any(args.pe)
    px_long = _read_any(args.prices)
    ind_df = _read_any(args.industry)
    reg_df = _read_any(args.regimes)

    pe_w, px_w = _prepare_monthly_panels(pe_long, px_long)
    # Month-start open panel for open-to-open forward returns (required)
    mo = _read_any(args.prices_monthstart_open)
    if not {"month", "symbol", "open"}.issubset(set(mo.columns)):
        raise SystemExit("--prices-monthstart-open must contain columns: month,symbol,open (date optional).")
    mo = mo.copy()
    mo["month"] = mo["month"].astype(str)
    mo["symbol"] = mo["symbol"].astype(str).str.strip()
    mo["open"] = pd.to_numeric(mo["open"], errors="coerce")
    # Map month -> month-start open, index it by month-end timestamps (same as pe_w/px_w index)
    mo["month_end"] = pd.PeriodIndex(mo["month"], freq="M").to_timestamp("M")
    open_w = mo.pivot(index="month_end", columns="symbol", values="open").sort_index()
    # align to pe/px panels intersection
    open_w = open_w.reindex(index=pe_w.index, columns=pe_w.columns)

    # industry map
    ind_df = ind_df.rename(columns={"ts_code": "symbol"}).copy()
    ind_df["symbol"] = ind_df["symbol"].astype(str).str.strip()
    ind_df["industry"] = ind_df["industry"].astype(str).str.strip()
    industry_map = ind_df.drop_duplicates("symbol").set_index("symbol")["industry"]

    # regimes by month
    reg = reg_df.copy()
    if "month" not in reg.columns:
        raise SystemExit("regimes file must contain 'month' column (YYYY-MM).")
    reg["month"] = reg["month"].astype(str)
    reg = reg.set_index("month")

    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    floor = float(args.floor)
    min_names = int(args.min_names)

    # Precompute momentum for threshold variant
    mom = _momentum(px_w, lookback_m=int(args.mom_lookback_m))

    results_series = []
    results_summary = []

    for h in horizons:
        # open-to-open using month-start opens
        fwd = open_w.shift(-h) / open_w - 1.0

        # build scores per variant
        scores: dict[Variant, pd.DataFrame] = {}
        scores["plain"] = (-pe_w).astype("float64")

        eligible = (mom > 0) & pe_w.notna() & (pe_w > 0) & np.isfinite(pe_w)
        scores["threshold"] = (-pe_w).where(eligible, other=floor).astype("float64")

        ind_scores = []
        for d in pe_w.index:
            ind_scores.append(_industry_neutral_score(pe_w.loc[d], industry_map).rename(d))
        scores["industry_neutral"] = pd.DataFrame(ind_scores)
        scores["industry_neutral"].index = pe_w.index
        scores["industry_neutral"] = scores["industry_neutral"].reindex(columns=pe_w.columns)

        for variant, sc in scores.items():
            ic = _compute_ic_series(sc, fwd, min_names=min_names)

            # attach to long series output
            tmp = pd.DataFrame({"date": ic.index, "horizon_m": h, "variant": variant, "ic": ic.values})
            results_series.append(tmp)

            # summaries overall + by regimes
            summ_all = _summarize_ic(ic)
            results_summary.append(
                {
                    "horizon_m": h,
                    "variant": variant,
                    "regime": "all",
                    **summ_all,
                }
            )

            # by market states
            months = ic.index.to_period("M").astype(str)
            ic_by_month = pd.Series(ic.values, index=months)
            joined = pd.DataFrame({"ic": ic_by_month}).join(reg[["up", "high_vol"]], how="left")

            def _summ(name: str, mask: pd.Series) -> None:
                s = joined.loc[mask, "ic"]
                results_summary.append({"horizon_m": h, "variant": variant, "regime": name, **_summarize_ic(s)})

            _summ("up", joined["up"] == True)  # noqa: E712
            _summ("down", joined["up"] == False)  # noqa: E712
            _summ("high_vol", joined["high_vol"] == True)  # noqa: E712
            _summ("low_vol", joined["high_vol"] == False)  # noqa: E712
            _summ("up_high_vol", (joined["up"] == True) & (joined["high_vol"] == True))  # noqa: E712
            _summ("up_low_vol", (joined["up"] == True) & (joined["high_vol"] == False))  # noqa: E712
            _summ("down_high_vol", (joined["up"] == False) & (joined["high_vol"] == True))  # noqa: E712
            _summ("down_low_vol", (joined["up"] == False) & (joined["high_vol"] == False))  # noqa: E712

    ic_series = pd.concat(results_series, ignore_index=True)
    ic_summary = pd.DataFrame(results_summary)

    _write_any(ic_series, out_dir / "ic_series.parquet")
    _write_any(ic_summary, out_dir / "ic_summary.parquet")

    print(f"Wrote: {out_dir / 'ic_series.parquet'} rows={len(ic_series):,}")
    print(f"Wrote: {out_dir / 'ic_summary.parquet'} rows={len(ic_summary):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

