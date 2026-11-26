"""Compare streaming output (run with --emit-all) against the batch reference.

Usage: python scripts/validate.py results/runs/X/windows.parquet results/reference.parquet
Reports: missing/extra (symbol, window) rows, max abs/rel EMA difference, advisory agreement.
"""
import json
import sys

import polars as pl


def validate(stream_path, ref_path, rel_tol=1e-9) -> dict:
    s = pl.read_parquet(stream_path).select(["symbol", "win", "close", "ema38", "ema100", "advice"])
    r = pl.read_parquet(ref_path).select(["symbol", "win", "close", "ema38", "ema100", "advice"])
    j = s.join(r, on=["symbol", "win"], how="full", suffix="_ref", coalesce=True)
    missing = j.filter(pl.col("close").is_null()).height       # in reference, not in stream
    extra = j.filter(pl.col("close_ref").is_null()).height     # in stream, not in reference
    both = j.drop_nulls(["close", "close_ref"])

    def rel(c):
        return ((pl.col(c) - pl.col(f"{c}_ref")).abs() / pl.col(f"{c}_ref").abs().clip(1e-12)).max()

    diffs = both.select(
        close_mismatch=(pl.col("close") != pl.col("close_ref")).sum(),
        ema38_max_rel=rel("ema38"), ema100_max_rel=rel("ema100"),
        advice_mismatch=(pl.col("advice") != pl.col("advice_ref")).sum(),
    ).row(0, named=True)
    rep = dict(
        stream_rows=s.height, reference_rows=r.height, matched_rows=both.height,
        missing_in_stream=missing, extra_in_stream=extra, **diffs,
        stream_buy=int((s["advice"] == 1).sum()), stream_sell=int((s["advice"] == -1).sum()),
        ref_buy=int((r["advice"] == 1).sum()), ref_sell=int((r["advice"] == -1).sum()),
    )
    rep["pass"] = (missing == 0 and extra == 0 and diffs["close_mismatch"] == 0
                   and diffs["advice_mismatch"] == 0
                   and (diffs["ema38_max_rel"] or 0) <= rel_tol and (diffs["ema100_max_rel"] or 0) <= rel_tol)
    return rep


if __name__ == "__main__":
    rep = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=2, default=float))
    sys.exit(0 if rep["pass"] else 1)
