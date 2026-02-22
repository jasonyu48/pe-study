from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pe_research import (
    _prepare_monthly_panels,
    _read_any,
    _simulate_drift_rebalance,
    _summarize_performance,
    _write_any,
)


def _zscore_rowwise(x: pd.DataFrame, *, fill_missing: float | None) -> pd.DataFrame:
    x = x.replace([np.inf, -np.inf], np.nan).astype("float64")
    if fill_missing is not None:
        x = x.fillna(float(fill_missing))
    mean = x.mean(axis=1, skipna=True)
    std = x.std(axis=1, ddof=1, skipna=True).replace(0.0, np.nan)
    z = x.sub(mean, axis=0).div(std, axis=0)
    # If std is nan (e.g., all equal), treat as 0 for all names (no cross-sectional info).
    return z.fillna(0.0).astype("float64")


def _lambda_grid(lam_min: float, lam_max: float, step: float) -> list[float]:
    if step <= 0:
        raise SystemExit("--lambda-step must be positive.")
    n_steps = int(np.floor((lam_max - lam_min) / step + 1e-9)) + 1
    return [float(np.round(lam_min + i * step, 10)) for i in range(n_steps)]


def _plot_heatmap(mat: pd.DataFrame, *, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = mat.copy()
    # rows: mom lookback, cols: lambda
    m = m.sort_index(axis=0).sort_index(axis=1)
    arr = m.to_numpy(dtype="float64")

    fig = plt.figure(figsize=(12, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(arr, aspect="auto", origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe (ann., monthly)")

    ax.set_title(title)
    ax.set_xlabel("lambda")
    ax.set_ylabel("Momentum lookback (months)")

    # ticks
    xt = list(range(m.shape[1]))
    yt = list(range(m.shape[0]))
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([str(c) for c in m.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in m.index])

    # annotate best
    if np.isfinite(arr).any():
        best_pos = np.nanargmax(arr)
        by, bx = np.unravel_index(best_pos, arr.shape)
        ax.scatter([bx], [by], s=40, c="white", marker="o", edgecolors="black", linewidths=0.6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heatmap study: Sharpe over (lambda, momentum lookback) grid for one horizon h.")
    p.add_argument("--pe", type=str, default="data/cn_pe_monthly.parquet")
    p.add_argument("--prices", type=str, default="data/cn_prices_monthend.parquet", help="Month-end prices (should include adj_close).")
    p.add_argument("--prices-monthstart-open", type=str, default="data/cn_prices_monthstart_open.parquet")
    p.add_argument("--out-dir", type=str, default="reports/lambda_mom_heatmap")

    p.add_argument("--horizon-m", type=int, default=6)
    p.add_argument("--top-quantile", type=float, default=0.2)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--max-stale-months", type=int, default=3)

    p.add_argument("--floor", type=float, default=-1e12, help="Floor score for invalid PE (to match plain).")
    p.add_argument("--mom-min", type=int, default=1)
    p.add_argument("--mom-max", type=int, default=12)

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

    mo = _read_any(args.prices_monthstart_open)
    if not {"month", "symbol", "open"}.issubset(set(mo.columns)):
        raise SystemExit("--prices-monthstart-open must contain columns: month,symbol,open")
    mo = mo.copy()
    mo["month"] = mo["month"].astype(str)
    mo["symbol"] = mo["symbol"].astype(str).str.strip()
    mo["open"] = pd.to_numeric(mo["open"], errors="coerce")
    mo["month_end"] = pd.PeriodIndex(mo["month"], freq="M").to_timestamp("M")
    open_w = mo.pivot(index="month_end", columns="symbol", values="open").sort_index()

    # align intersection
    open_w = open_w.reindex(index=pe_w.index, columns=pe_w.columns)

    floor = float(args.floor)
    valid_pe = pe_w.notna() & (pe_w > 0) & np.isfinite(pe_w)
    val_raw = (-pe_w).where(valid_pe, other=floor).astype("float64")
    z_val = _zscore_rowwise(val_raw, fill_missing=floor)

    lam_list = _lambda_grid(float(args.lambda_min), float(args.lambda_max), float(args.lambda_step))

    mom_min = int(args.mom_min)
    mom_max = int(args.mom_max)
    if mom_min <= 0 or mom_max < mom_min:
        raise SystemExit("--mom-min must be >=1 and --mom-max must be >= mom-min.")
    mom_list = list(range(mom_min, mom_max + 1))

    rows = []
    # main loop: compute z_mom once per lookback, then sweep lambdas
    for m in mom_list:
        mom_raw = px_w / px_w.shift(int(m)) - 1.0
        z_mom = _zscore_rowwise(mom_raw, fill_missing=None)  # missing -> neutral after zscore fill

        for lam in lam_list:
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
            rows.append(
                {
                    "horizon_m": int(args.horizon_m),
                    "mom_lookback_m": int(m),
                    "lambda": float(lam),
                    "sharpe": sharpe,
                    "cagr": float(perf.get("cagr", np.nan)),
                    "total_return": float(perf.get("total_return", np.nan)),
                    "max_drawdown": float(perf.get("max_drawdown", np.nan)),
                }
            )

    res = pd.DataFrame(rows)
    _write_any(res, out_dir / "heatmap_grid.parquet")
    _write_any(res, out_dir / "heatmap_grid.csv")

    mat = res.pivot(index="mom_lookback_m", columns="lambda", values="sharpe").sort_index()
    _write_any(mat.reset_index(), out_dir / "heatmap_matrix.csv")

    png = out_dir / "sharpe_heatmap.png"
    title = f"Sharpe heatmap (h={int(args.horizon_m)}m): score=z(-PE)+lambda*z(Mom)"
    _plot_heatmap(mat, out_path=png, title=title)

    # best cell
    best = res.loc[res["sharpe"].astype(float).idxmax()]
    md = []
    md.append("## Lambda × Momentum-lookback heatmap\n\n")
    md.append(f"- Horizon (rebalance): **{int(args.horizon_m)}m**\n")
    md.append(f"- Lambda grid: [{float(args.lambda_min)}, {float(args.lambda_max)}] step={float(args.lambda_step)}\n")
    md.append(f"- Mom lookback grid: [{mom_min}, {mom_max}] (months)\n")
    md.append(f"- Best: mom={int(best['mom_lookback_m'])}m, lambda={float(best['lambda'])}, Sharpe={float(best['sharpe']):.3f}\n\n")
    md.append("![Sharpe heatmap](sharpe_heatmap.png)\n")
    (out_dir / "HEATMAP_REPORT.md").write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_dir / 'heatmap_grid.parquet'} rows={len(res)}")
    print(f"Wrote: {png}")
    print(f"Wrote: {out_dir / 'HEATMAP_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

