"""Independent batch reference implementation of Query 1 and Query 2 (for validation, D12).

Deliberately implemented differently from the streaming engine:
  * polars (not pyarrow/numpy), whole dataset at once, no watermarks (sees all events),
  * Close = last price per (symbol, window) via sort + group_by.last,
  * EMA via polars' ewm_mean with a closed-form correction for EMA_0 = 0:
      ewm_mean(adjust=False) starts with y_0 = x_0; the spec's EMA starts from 0.
      With e_n the spec EMA: y_n - e_n = (1-a)^(n+1) * x_0, so e_n = y_n - (1-a)^(n+1) * x_0.
  * crossovers from shifted columns, warm-up K applied as "n >= K previous windows".

Usage: python scripts/reference.py FILES... --out results/reference.parquet [--max-rows N] [--warmup K]
"""
import argparse
import glob
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tickstream.schema import read_csv_lazy  # noqa: E402

W = 5 * 60 * 1000


def reference(files, max_rows=None, warmup=38) -> pl.DataFrame:
    lf = pl.concat([read_csv_lazy(f) for f in files])
    if max_rows:
        lf = lf.head(max_rows)
    ev = (
        lf.with_row_index("row")
        .with_columns(price=pl.col("Last").cast(pl.Float64, strict=False))
        .filter((pl.col("price") > 0) & pl.col("Trading time").is_not_null())
        .with_columns(
            t=pl.concat_str([pl.coalesce("Trading date", "Date"), pl.lit(" "), pl.col("Trading time")])
            .str.strptime(pl.Datetime("ms"), "%d-%m-%Y %H:%M:%S%.f")
            .dt.epoch("ms")
        )
        .with_columns(win=pl.col("t") // W)
    )
    closes = (
        ev.sort(["ID", "win", "t", "row"])
        .group_by(["ID", "win"], maintain_order=True)
        .agg(close=pl.col("price").last())
        .sort(["ID", "win"])
        .collect(engine="streaming")
    )
    out = closes.with_columns(n=pl.int_range(pl.len()).over("ID"),
                              x0=pl.col("close").first().over("ID"))
    for j in (38, 100):
        a = 2.0 / (1 + j)
        out = out.with_columns(
            (pl.col("close").ewm_mean(alpha=a, adjust=False).over("ID")
             - (1 - a) ** (pl.col("n") + 1) * pl.col("x0")).alias(f"ema{j}")
        )
    out = out.with_columns(
        prev38=pl.col("ema38").shift(1, fill_value=0.0).over("ID"),
        prev100=pl.col("ema100").shift(1, fill_value=0.0).over("ID"),
    )
    eligible = pl.col("n") >= warmup
    buy = (pl.col("ema38") > pl.col("ema100")) & (pl.col("prev38") <= pl.col("prev100")) & eligible
    sell = (pl.col("ema38") < pl.col("ema100")) & (pl.col("prev38") >= pl.col("prev100")) & eligible
    out = out.with_columns(
        advice=pl.when(buy).then(1).when(sell).then(-1).otherwise(0).cast(pl.Int8)
    ).rename({"ID": "symbol"})
    return out.select(["symbol", "win", "close", "ema38", "ema100", "prev38", "prev100", "advice"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/reference.parquet")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=38)
    a = ap.parse_args()
    files = [g for f in a.files for g in (sorted(glob.glob(f)) or [f])]
    df = reference(files, a.max_rows, a.warmup)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out)
    print(f"{df.height} (symbol, window) rows, {df['symbol'].n_unique()} symbols, "
          f"{(df['advice'] == 1).sum()} BUY, {(df['advice'] == -1).sum()} SELL -> {a.out}")


if __name__ == "__main__":
    main()
