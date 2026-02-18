from __future__ import annotations

from pathlib import Path


def read_tushare_token(path: str | Path = "data/token") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TuShare token file not found: {p}")
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("TuShare token file is empty.")
    return token


def get_pro(token_path: str | Path = "data/token"):
    """
    Return a TuShare PRO client using token stored in `data/token`.
    """
    try:
        import tushare as ts
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency. Install with: pip install tushare") from e

    token = read_tushare_token(token_path)
    ts.set_token(token)
    return ts.pro_api(token)

