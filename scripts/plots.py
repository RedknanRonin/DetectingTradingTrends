"""Report figures from benchmark summaries, profiling output and a run directory.

Usage:
  python scripts/plots.py --bench results/bench/<tag>/summary.csv [more.csv ...]
                          --profile results/profile --run results/runs/week --symbol AIR.FR
Figures go to results/figures/*.pdf and *.png (single-column ACM width).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

OUT = Path("results/figures")
plt.rcParams.update({"figure.figsize": (3.4, 2.4), "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight", "legend.fontsize": 7})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def agg(df, by, col):
    return df.group_by(by).agg(pl.col(col).mean().alias("mean"), pl.col(col).std().alias("std"),
                               pl.col(col).min().alias("min"), pl.col(col).max().alias("max")).sort(by)


def fig_scaling(b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.4))
    for exp, label, m in (("cores", "end-to-end", "o-"), ("preload", "processing only (preloaded)", "s--")):
        d = b.filter((pl.col("experiment") == exp) & (pl.col("partitioning") == "hash"))
        if d.is_empty():
            continue
        a = agg(d, "cores", "price_events_per_s")
        ax1.errorbar(a["cores"], a["mean"] / 1e6, yerr=a["std"].fill_null(0) / 1e6, fmt=m, label=label, capsize=2)
        base = a["mean"][0]
        ax2.plot(a["cores"], a["mean"] / base, m, label=label)
    if ax2.lines:
        c = b["cores"].unique().sort()
        ax2.plot(c, c / c.min(), ":", color="grey", label="ideal")
    ax1.set(xlabel="cores", ylabel="throughput (M price events/s)")
    ax2.set(xlabel="cores", ylabel="speed-up vs. 1 core")
    ax1.legend(); ax2.legend()
    save(fig, "scaling")


def fig_partition(b):
    d = b.filter(pl.col("experiment") == "partition")
    if d.is_empty():
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.4))
    for p, m in (("hash", "o-"), ("balanced", "s--")):
        x = d.filter(pl.col("partitioning") == p)
        a = agg(x, "cores", "load_imbalance")
        ax1.plot(a["cores"], a["mean"], m, label=p)
        a = agg(x, "cores", "price_events_per_s")
        ax2.errorbar(a["cores"], a["mean"] / 1e6, yerr=a["std"].fill_null(0) / 1e6, fmt=m, label=p, capsize=2)
    ax1.set(xlabel="workers (= cores)", ylabel="load imbalance (max/mean events)")
    ax2.set(xlabel="workers (= cores)", ylabel="throughput (M price events/s)")
    ax1.legend(); ax2.legend()
    save(fig, "partitioning")


def fig_lateness(b):
    d = b.filter(pl.col("experiment") == "lateness")
    if d.is_empty():
        return
    a = agg(d, "lateness_ms", "late_dropped")
    l = agg(d, "lateness_ms", "lat_p50_ms")
    fig, ax = plt.subplots()
    x = [str(v) for v in a["lateness_ms"]]
    ax.bar(x, a["mean"], color="#d73027", alpha=0.7)
    ax.set(xlabel="allowed lateness (ms)", ylabel="late events dropped")
    ax2 = ax.twinx()
    ax2.plot(x, l["mean"], "o-", color="#2c7bb6")
    ax2.set_ylabel("window latency p50 (ms)", color="#2c7bb6")
    ax2.grid(False)
    save(fig, "lateness")


def fig_block(b):
    d = b.filter(pl.col("experiment") == "block")
    if d.is_empty():
        return
    t = agg(d, "block_mb", "rows_per_s")
    p50 = agg(d, "block_mb", "lat_p50_ms")
    p99 = agg(d, "block_mb", "lat_p99_ms")
    fig, ax = plt.subplots()
    x = [f"{v:g}" for v in t["block_mb"]]
    ax.plot(x, t["mean"] / 1e6, "o-", color="black", label="throughput")
    ax.set(xlabel="source block size (MB)", ylabel="throughput (M events/s)")
    ax2 = ax.twinx()
    ax2.plot(x, p50["mean"], "s--", color="#2c7bb6", label="latency p50")
    ax2.plot(x, p99["mean"], "^--", color="#d73027", label="latency p99")
    ax2.set_ylabel("window latency (ms)")
    ax2.grid(False)
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.88))
    save(fig, "blocksize")


def fig_profile(prof: Path):
    rate = prof / "price_events_per_minute.csv"
    if rate.exists():
        r = pl.read_csv(rate).with_columns(ts=pl.from_epoch("minute", time_unit="ms"))
        fig, ax = plt.subplots(figsize=(7, 2.0))
        ax.plot(r["ts"], r["len"] / 60, lw=0.5)
        ax.set(ylabel="price events / s", xlabel="trading time (minute averages)")
        save(fig, "workload_rate")
    per = prof / "price_events_per_symbol.csv"
    if per.exists():
        p = pl.read_csv(per).sort("len", descending=True)
        share = (p["len"].cum_sum() / p["len"].sum()).to_list()
        fig, ax = plt.subplots()
        ax.plot(range(1, len(share) + 1), share)
        ax.set_xscale("log")
        ax.set(xlabel="top-k symbols (log)", ylabel="share of price events")
        save(fig, "long_tail")


def fig_symbol(run: Path, symbol: str):
    w = pl.read_parquet(run / "windows.parquet").filter(pl.col("symbol") == symbol).sort("win")
    if w.is_empty():
        print("symbol not in run:", symbol)
        return
    fig, ax = plt.subplots(figsize=(7, 2.4))
    x = range(w.height)  # window index (compresses nights/weekends)
    ax.plot(x, w["close"], color="grey", lw=0.7, label="Close")
    ax.plot(x, w["ema38"], color="#2c7bb6", label="EMA38")
    ax.plot(x, w["ema100"], color="#fdae61", label="EMA100")
    idx = list(x)
    for code, mk, c, lab in ((1, "^", "#1a9850", "BUY"), (-1, "v", "#d73027", "SELL")):
        sel = [i for i, a in zip(idx, w["advice"]) if a == code]
        ax.scatter(sel, [w["close"][i] for i in sel], marker=mk, color=c, s=40, zorder=5, label=lab)
    days = w["window_start"].dt.date()
    ticks = [i for i in range(1, w.height) if days[i] != days[i - 1]]
    ax.set_xticks(ticks, [f"{days[i]:%a}" for i in ticks])
    ax.set(ylabel="price", title=symbol)
    ax.legend(ncol=5, loc="lower right")
    save(fig, f"symbol_{symbol.replace('.', '_')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--symbol", nargs="*", default=[])
    a = ap.parse_args()
    if a.bench:
        b = pl.concat([pl.read_csv(f) for f in a.bench], how="diagonal_relaxed")
        fig_scaling(b); fig_partition(b); fig_lateness(b); fig_block(b)
    if a.profile:
        fig_profile(Path(a.profile))
    if a.run:
        for s in a.symbol:
            fig_symbol(Path(a.run), s)


if __name__ == "__main__":
    main()
