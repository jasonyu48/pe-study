from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Combo:
    horizon_m: int
    variant: str


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return pd.read_parquet(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(x: float, *, digits: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return ""
    return f"{float(x):.{digits}f}"


def _fmt_pct(x: float, *, digits: int = 1) -> str:
    if x is None or not np.isfinite(float(x)):
        return ""
    return f"{100.0 * float(x):.{digits}f}%"


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    # Minimal pipe table generator (avoids pandas.to_markdown dependency on tabulate).
    if df.empty:
        return "_(no rows)_\n"

    def as_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(float(v)):
                return ""
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[as_str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(x) for x in col) for col in cols]

    def fmt_row(r):
        return "| " + " | ".join(s.ljust(w) for s, w in zip(r, widths)) + " |"

    out = []
    out.append(fmt_row(headers))
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out) + "\n"


def _load_ic_all(ic_summary: pd.DataFrame) -> pd.DataFrame:
    s = ic_summary.copy()
    if "regime" not in s.columns:
        raise SystemExit("ic_summary.parquet must contain 'regime' column.")
    s = s.loc[s["regime"].astype(str).str.lower() == "all"].copy()
    keep = ["horizon_m", "variant", "mean_ic", "ic_ir", "t_stat", "n"]
    for c in keep:
        if c not in s.columns:
            raise SystemExit(f"ic_summary.parquet missing column: {c}")
    return s[keep].copy()


def _discover_backtest_summaries(run_dir: Path) -> list[tuple[Combo, Path, dict]]:
    out = []
    root = run_dir / "backtests"
    if not root.exists():
        raise SystemExit(f"Missing backtests folder: {root}")
    for summary_path in root.glob("h*m/*/summary.json"):
        js = _read_json(summary_path)
        h = int(js["horizon_m"])
        v = str(js["variant"])
        out.append((Combo(h, v), summary_path, js))
    if not out:
        raise SystemExit(f"No backtest summaries found under: {root}")
    return out


def _equity_curve_path(run_dir: Path, combo: Combo) -> Path:
    return run_dir / "backtests" / f"h{combo.horizon_m}m" / combo.variant / "equity_curve.parquet"


def _pick_best(table: pd.DataFrame) -> Combo:
    if table.empty:
        raise SystemExit("No combos available to select best.")
    t = table.copy()
    # sort desc by sharpe; tie-breaker by total_return, then smaller drawdown (less negative).
    t["max_drawdown"] = pd.to_numeric(t["max_drawdown"], errors="coerce")
    t["sharpe"] = pd.to_numeric(t["sharpe"], errors="coerce")
    t["total_return"] = pd.to_numeric(t["total_return"], errors="coerce")
    t = t.sort_values(["sharpe", "total_return", "max_drawdown"], ascending=[False, False, False], kind="mergesort")
    row = t.iloc[0]
    return Combo(int(row["horizon_m"]), str(row["variant"]))


def _plot_best_equity(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if "date" not in d.columns or "equity" not in d.columns:
        raise SystemExit("equity_curve.parquet must contain columns: date,equity")
    d["date"] = pd.to_datetime(d["date"])
    d["equity"] = pd.to_numeric(d["equity"], errors="coerce")
    d = d.dropna(subset=["date", "equity"]).sort_values("date")
    if d.empty:
        raise SystemExit("equity curve is empty after cleaning.")

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["date"], d["equity"], linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate markdown report from pe_research outputs.")
    p.add_argument("--run-dir", type=str, required=True, help="Run output dir produced by src.pe_research (contains ic_summary.parquet + backtests/)")
    p.add_argument("--out-md", type=str, default="", help="Output markdown path (default: <run-dir>/REPORT.md)")
    p.add_argument("--top-n", type=int, default=20, help="Show top N combos in the table (sorted by Sharpe). Use 0 to show all.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    out_md = Path(args.out_md) if str(args.out_md).strip() else (run_dir / "REPORT.md")

    ic_summary = _read_parquet(run_dir / "ic_summary.parquet")
    ic_all = _load_ic_all(ic_summary)

    bt_items = _discover_backtest_summaries(run_dir)
    bt_rows = []
    for combo, _, js in bt_items:
        eq_path = _equity_curve_path(run_dir, combo)
        eq = _read_parquet(eq_path)
        avg_n_holdings = float(pd.to_numeric(eq.get("n_holdings"), errors="coerce").mean())
        bt_rows.append(
            {
                "horizon_m": combo.horizon_m,
                "variant": combo.variant,
                "sharpe": js.get("sharpe", np.nan),
                "total_return": js.get("total_return", np.nan),
                "cagr": js.get("cagr", np.nan),
                "max_drawdown": js.get("max_drawdown", np.nan),
                "n_periods": js.get("n_periods", np.nan),
                "avg_n_holdings": avg_n_holdings,
            }
        )
    bt_df = pd.DataFrame(bt_rows)

    merged = bt_df.merge(ic_all, on=["horizon_m", "variant"], how="left")
    merged["sharpe"] = pd.to_numeric(merged["sharpe"], errors="coerce")
    merged = merged.sort_values(["sharpe", "total_return"], ascending=[False, False], kind="mergesort")

    best = _pick_best(merged)
    best_curve = _read_parquet(_equity_curve_path(run_dir, best))
    best_png = run_dir / "best_equity_curve.png"
    best_title = f"Best by Sharpe: h={best.horizon_m}m, {best.variant}"
    _plot_best_equity(best_curve, out_path=best_png, title=best_title)

    # pretty table
    show = merged.copy()
    show["Sharpe"] = show["sharpe"].map(lambda x: _fmt_float(x, digits=3))
    show["IC(mean)"] = show["mean_ic"].map(lambda x: _fmt_float(x, digits=4))
    show["Return(total)"] = show["total_return"].map(lambda x: _fmt_pct(x, digits=1))
    show["CAGR"] = show["cagr"].map(lambda x: _fmt_pct(x, digits=1))
    show["MaxDD"] = show["max_drawdown"].map(lambda x: _fmt_pct(x, digits=1))
    show["AvgHold"] = show["avg_n_holdings"].map(lambda x: _fmt_float(x, digits=0))
    show["horizon_m"] = show["horizon_m"].astype(int)
    show["variant"] = show["variant"].astype(str)
    show = show[["horizon_m", "variant", "Sharpe", "IC(mean)", "Return(total)", "CAGR", "MaxDD", "AvgHold", "n_periods"]]
    show = show.rename(columns={"horizon_m": "h(m)", "variant": "variant", "n_periods": "N"})
    if int(args.top_n) > 0:
        show = show.head(int(args.top_n))

    md = []
    md.append("## PE study — backtest + IC report\n")
    md.append(f"- **Run dir**: `{run_dir.as_posix()}`\n")
    md.append(f"- **Best combo (by Sharpe)**: **h={best.horizon_m}m**, **{best.variant}**\n")
    md.append("\n")
    md.append("## All combinations (sorted by Sharpe)\n\n")
    md.append(_df_to_markdown_table(show))
    md.append("\n")
    md.append("## Best combination — equity curve\n\n")
    md.append(f"![best equity curve]({best_png.name})\n")

    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {best_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

