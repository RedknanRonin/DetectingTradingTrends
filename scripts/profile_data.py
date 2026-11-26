"""Dataset profiling for the DEBS 2022 Trading Data.

Answers the questions that drive the design decisions:
  * how many events are price events (Last present / Last > 0)?
  * distribution over exchanges, security types, symbols (long tail)
  * how far does Trading time lag behind the system arrival Time?
  * how out-of-order is Trading time in file order (globally and per symbol)?
    -> needed to choose the watermark lateness (D5)

Usage: python scripts/profile_data.py FILE [FILE ...] [--out results/profile]
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tickstream.schema import read_csv_lazy  # noqa: E402


def to_ms(date_col: str | pl.Expr, time_col: str) -> pl.Expr:
    """DD-MM-YYYY + HH:MM:SS.fff -> epoch milliseconds (naive, CEST wall clock)."""
    date = pl.col(date_col) if isinstance(date_col, str) else date_col
    return (
        pl.concat_str([date, pl.lit(" "), pl.col(time_col)])
        .str.strptime(pl.Datetime("ms"), "%d-%m-%Y %H:%M:%S%.f", strict=False)
        .dt.epoch("ms")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/profile")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Memory-safe for the full week (289M rows): aggregates over ALL events are computed as
    # streaming lazy queries; only the (much smaller) price events are materialised, with
    # a minimal set of columns.
    lf = (
        pl.concat([read_csv_lazy(f) for f in args.files])
        .with_row_index("row")
        .with_columns(
            last=pl.col("Last").cast(pl.Float64, strict=False),
            exchange=pl.col("ID").str.extract(r"\.([A-Z]+)$"),
        )
    )

    def stream(q):
        return q.collect(engine="streaming")

    report: dict = {}
    totals = stream(lf.select(
        events_total=pl.len(),
        symbols_total=pl.col("ID").n_unique(),
        events_with_last=pl.col("last").is_not_null().sum(),
        events_last_zero=(pl.col("last") == 0).sum(),
        system_time_whole_seconds_frac=pl.col("Time").str.ends_with(".000").mean(),
    )).row(0, named=True)
    report.update(totals)
    report["by_exchange"] = dict(stream(lf.group_by("exchange").len()).iter_rows())
    report["by_sectype"] = dict(stream(lf.group_by("SecType").len()).iter_rows())
    all_sym = stream(lf.group_by("ID").len()).sort("len", descending=True)
    all_sym.write_csv(out / "all_events_per_symbol.csv")
    per_day = stream(lf.group_by("Date").agg(events=pl.len(), price_events=(pl.col("last") > 0).sum()))
    report["events_by_day"] = {d: (e, pe) for d, e, pe in per_day.sort("Date").iter_rows()}

    price = stream(
        lf.filter(pl.col("last") > 0)
        .with_columns(
            sys_ms=to_ms("Date", "Time"),
            # Trading date is mostly empty (filled for ~9% of price events, and then always
            # equal to Date), so the event date falls back to the system Date.
            trd_ms=to_ms(pl.coalesce("Trading date", "Date"), "Trading time"),
            has_trading_date=pl.col("Trading date").is_not_null(),
            td_mismatch=pl.col("Trading date").is_not_null() & (pl.col("Trading date") != pl.col("Date")),
        )
        .select(["row", "ID", "SecType", "exchange", "sys_ms", "trd_ms", "has_trading_date", "td_mismatch"])
    )
    report["events_price_gt0"] = price.height
    report["price_events_trading_time_missing"] = price.filter(pl.col("trd_ms").is_null()).height
    report["price_events_by_sectype"] = dict(price.group_by("SecType").len().iter_rows())
    report["price_events_by_exchange"] = dict(price.group_by("exchange").len().iter_rows())
    report["symbols_with_price"] = price["ID"].n_unique()
    report["price_events_with_trading_date"] = int(price["has_trading_date"].sum())
    report["trading_date_vs_date_mismatch"] = int(price["td_mismatch"].sum())

    # --- long tail: share of price events covered by the top-k symbols
    per_sym = price.group_by("ID").len().sort("len", descending=True)
    per_sym.write_csv(out / "price_events_per_symbol.csv")
    tot = per_sym["len"].sum()
    cum = per_sym["len"].cum_sum()
    report["top_k_share_price_events"] = {
        k: round(float(cum[min(k, len(cum)) - 1]) / tot, 4) for k in (1, 10, 50, 100, 500, 1000)
    }
    report["price_events_per_symbol_quantiles"] = {
        q: float(per_sym["len"].quantile(q)) for q in (0.5, 0.9, 0.99)
    }
    report["price_events_per_symbol_max"] = int(per_sym["len"].max())

    # --- delay between trade time and system arrival time (price events)
    d = price.select((pl.col("sys_ms") - pl.col("trd_ms")).alias("d"))["d"].drop_nulls()
    report["arrival_minus_trading_ms_quantiles"] = {
        q: float(d.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
    }

    # --- disorder of event time in file order (price events only)
    p = price.filter(pl.col("trd_ms").is_not_null()).sort("row")
    p = p.with_columns(
        g_late=(pl.col("trd_ms").cum_max() - pl.col("trd_ms")),
        s_late=(pl.col("trd_ms").cum_max().over("ID") - pl.col("trd_ms")),
    )
    thresholds = [0, 100, 1_000, 5_000, 30_000, 60_000, 300_000]
    report["global_lateness_frac_gt_ms"] = {
        t: round(p.filter(pl.col("g_late") > t).height / p.height, 6) for t in thresholds
    }
    report["per_symbol_lateness_frac_gt_ms"] = {
        t: round(p.filter(pl.col("s_late") > t).height / p.height, 6) for t in thresholds
    }
    report["global_lateness_ms_quantiles"] = {
        q: float(p["g_late"].quantile(q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
    }
    report["global_lateness_ms_max"] = int(p["g_late"].max())

    # --- event rate over time (per minute, by trading time) for the workload plot
    rate = (
        p.with_columns(minute=(pl.col("trd_ms") // 60_000) * 60_000)
        .group_by("minute")
        .len()
        .sort("minute")
    )
    rate.write_csv(out / "price_events_per_minute.csv")
    report["peak_price_events_per_minute"] = int(rate["len"].max())

    (out / "profile.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
