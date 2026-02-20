from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pe_research import _prepare_monthly_panels, _read_any, _simulate_drift_rebalance, _summarize_performance, _write_any


def _zscore_rowwise(x: pd.DataFrame, *, min_names: int, fill_missing: float | None = None) -> pd.DataFrame:
    x = x.replace([np.inf, -np.inf], np.nan).astype("float64")
    if fill_missing is not None:
        x = x.fillna(float(fill_missing))
    cnt = x.notna().sum(axis=1)
    mean = x.mean(axis=1, skipna=True)
    std = x.std(axis=1, ddof=1, skipna=True).replace(0.0, np.nan)
    z = x.sub(mean, axis=0).div(std, axis=0)
    z = z.where(cnt >= int(min_names))
    return z.fillna(0.0).astype("float64")


def _plot_sharpe_vs_lambda(df: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["lambda"] = pd.to_numeric(d["lambda"], errors="coerce")
    d["sharpe"] = pd.to_numeric(d["sharpe"], errors="coerce")
    d = d.dropna(subset=["lambda", "sharpe"]).sort_values("lambda")
    if d.empty:
        raise SystemExit("No sharpe results to plot.")

    best = d.iloc[d["sharpe"].argmax()]
    best_lam = float(best["lambda"])
    best_sh = float(best["sharpe"])

    fig = plt.figure(figsize=(9, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["lambda"], d["sharpe"], marker="o", markersize=3.0, linewidth=1.2)
    ax.axvline(best_lam, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(f"Sharpe vs lambda (best lambda={best_lam:.1f}, sharpe={best_sh:.3f})")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Sharpe (ann., monthly)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lambda study: score=z(-PE)+lambda*z(Mom), backtest and plot Sharpe vs lambda.")
    p.add_argument("--pe", type=str, required=True)
    p.add_argument("--prices", type=str, required=True, help="Month-end prices (should include adj_close for qfq).")
    p.add_argument("--prices-monthstart-open", type=str, required=True, help="Month-start open (qfq-equivalent open).")
    p.add_argument("--out-dir", type=str, default="reports/lambda_study_h6")
    p.add_argument("--horizon-m", type=int, default=6, help="Rebalance every h months (fixed in this study).")
    p.add_argument("--mom-lookback-m", type=int, default=6, help="Momentum lookback in months.")
    p.add_argument("--top-quantile", type=float, default=0.2, help="Long top-quantile names by score.")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--max-stale-months", type=int, default=3)
    p.add_argument("--min-names", type=int, default=30, help="Min names per month to compute z-scores.")
    p.add_argument("--floor", type=float, default=-1e12, help="Floor score for invalid PE (to match pe_research plain).")
    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=2.0)
    p.add_argument("--lambda-step", type=float, default=0.1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pe_long = _read_any(args.pe)
    px_long = _read_any(args.prices)
    pe_w, px_w = _prepare_monthly_panels(pe_long, px_long)

    # month-start open (qfq-equivalent) panel indexed by month-end timestamps
    mo = _read_any(args.prices_monthstart_open)
    if not {"month", "symbol", "open"}.issubset(set(mo.columns)):
        raise SystemExit("--prices-monthstart-open must contain columns: month,symbol,open")
    mo = mo.copy()
    mo["month"] = mo["month"].astype(str)
    mo["symbol"] = mo["symbol"].astype(str).str.strip()
    mo["open"] = pd.to_numeric(mo["open"], errors="coerce")
    mo["month_end"] = pd.PeriodIndex(mo["month"], freq="M").to_timestamp("M")
    open_w = mo.pivot(index="month_end", columns="symbol", values="open").sort_index()

    # align panels intersection
    open_w = open_w.reindex(index=pe_w.index, columns=pe_w.columns)

    # build z(-PE) and z(Mom)
    valid_pe = pe_w.notna() & (pe_w > 0) & np.isfinite(pe_w)
    # Match pe_research plain: invalid PE gets a floor score, and top-quantile is computed over the full universe.
    floor = float(args.floor)
    val_raw = (-pe_w).where(valid_pe, other=floor).astype("float64")

    mom_raw = px_w / px_w.shift(int(args.mom_lookback_m)) - 1.0
    z_val = _zscore_rowwise(val_raw, min_names=int(args.min_names), fill_missing=floor)
    # For momentum, treat missing as neutral tilt (0) after standardization, to avoid shrinking the universe.
    z_mom = _zscore_rowwise(mom_raw, min_names=int(args.min_names), fill_missing=None)

    lam_min = float(args.lambda_min)
    lam_max = float(args.lambda_max)
    step = float(args.lambda_step)
    if step <= 0:
        raise SystemExit("--lambda-step must be positive.")
    # inclusive grid with stable rounding to 1 decimal by default
    n_steps = int(np.floor((lam_max - lam_min) / step + 1e-9)) + 1
    lambdas = [float(np.round(lam_min + i * step, 10)) for i in range(n_steps)]

    results = []
    best = {"lambda": None, "sharpe": -np.inf, "curve": None}

    for lam in lambdas:
        score = (z_val + float(lam) * z_mom).astype("float64")
        bt = _simulate_drift_rebalance(
            open_panel=open_w,
            score_panel=score,
            rebalance_every_m=int(args.horizon_m),
            top_quantile=float(args.top_quantile),
            cost_bps=float(args.cost_bps),
            max_stale_months=int(args.max_stale_months),
        )
        perf = _summarize_performance(pd.Series(bt["return"].values, index=pd.to_datetime(bt["date"])))
        sharpe = float(perf.get("sharpe", np.nan))
        avg_hold = float(pd.to_numeric(bt.get("n_holdings"), errors="coerce").mean())
        forced_total = int(pd.to_numeric(bt.get("n_forced_drops"), errors="coerce").fillna(0).sum())

        results.append(
            {
                "lambda": float(lam),
                "sharpe": sharpe,
                "cagr": float(perf.get("cagr", np.nan)),
                "total_return": float(perf.get("total_return", np.nan)),
                "max_drawdown": float(perf.get("max_drawdown", np.nan)),
                "avg_n_holdings": avg_hold,
                "forced_drops_total": forced_total,
            }
        )

        if np.isfinite(sharpe) and sharpe > float(best["sharpe"]):
            best = {"lambda": float(lam), "sharpe": sharpe, "curve": bt}

    res = pd.DataFrame(results).sort_values("lambda").reset_index(drop=True)
    _write_any(res, out_dir / "lambda_grid.parquet")
    _write_any(res, out_dir / "lambda_grid.csv")

    _plot_sharpe_vs_lambda(res, out_path=out_dir / "sharpe_vs_lambda.png")

    # save best equity curve too
    if best["curve"] is not None:
        _write_any(pd.DataFrame(best["curve"]), out_dir / "best_equity_curve.parquet")

    top = res.sort_values("sharpe", ascending=False).head(10)
    md = []
    md.append("## Lambda study (h=6m)\n\n")
    md.append("- Score: `z(-PE) + lambda * z(Mom)`\n")
    md.append(f"- Horizon (rebalance): **{int(args.horizon_m)}m**\n")
    md.append(f"- Mom lookback: **{int(args.mom_lookback_m)}m**\n")
    md.append(f"- Lambda grid: [{lam_min}, {lam_max}] step={step}\n")
    md.append(f"- Best lambda (by Sharpe): **{best['lambda']}**, Sharpe={best['sharpe']:.3f}\n\n")
    md.append("![Sharpe vs lambda](sharpe_vs_lambda.png)\n\n")
    md.append("### Top-10 lambdas by Sharpe\n\n")
    md.append(top.to_string(index=False))
    md.append("\n")
    (out_dir / "LAMBDA_REPORT.md").write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_dir / 'lambda_grid.parquet'} rows={len(res)}")
    print(f"Wrote: {out_dir / 'sharpe_vs_lambda.png'}")
    print(f"Wrote: {out_dir / 'LAMBDA_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

