"""CLI: python -m tickstream FILES... [options]"""
import argparse
import glob
import json

from .config import RunConfig
from .engine import run

# 12700H: logical CPUs 0-11 are the 6 P-cores (2 hyper-threads each), 12-19 the 8 E-cores.
# Pin to distinct physical cores first (one thread per P-core), then E-cores, then HT siblings.
CORE_ORDER = [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 1, 3, 5, 7, 9, 11]


def main():
    ap = argparse.ArgumentParser(prog="tickstream")
    ap.add_argument("files", nargs="+", help="CSV day files (globs allowed), processed in order")
    ap.add_argument("--out", default="results/runs/default")
    ap.add_argument("--workers", type=int, default=None, help="default: number of cores")
    ap.add_argument("--cores", type=int, default=None, help="pin run to this many cores")
    ap.add_argument("--partitioning", choices=["hash", "balanced"], default="hash")
    ap.add_argument("--weights", default="results/profile/price_events_per_symbol.csv")
    ap.add_argument("--lateness-ms", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=38)
    ap.add_argument("--subscriptions", default="config/subscriptions.txt")
    ap.add_argument("--emit-all", action="store_true")
    ap.add_argument("--block-mb", type=float, default=16)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--source-threads", type=int, default=None)
    ap.add_argument("--preload", action="store_true",
                    help="parse everything first, then stream from memory (processing-only)")
    ap.add_argument("--no-bars", action="store_true", help="do not build 1-minute OHLC bars")
    a = ap.parse_args()

    files = []
    for f in a.files:
        files += sorted(glob.glob(f)) or [f]
    cores = CORE_ORDER[: a.cores] if a.cores else None
    cfg = RunConfig(
        files=files, out_dir=a.out, workers=a.workers or (a.cores or 4), cores=cores,
        partitioning=a.partitioning, weights_file=a.weights, lateness_ms=a.lateness_ms,
        warmup_windows=a.warmup, subscriptions_file=a.subscriptions, emit_all=a.emit_all,
        block_size=int(a.block_mb * (1 << 20)), max_rows=a.max_rows, source_threads=a.source_threads,
        preload=a.preload, minute_bars=not a.no_bars,
    )
    m = run(cfg)
    keys = ["wall_s", "rows", "price_events", "throughput_rows_per_s", "throughput_price_events_per_s",
            "load_imbalance", "result_latency_s", "cpu_process_tree_pct"]
    print(json.dumps({k: m[k] for k in keys} | {"late_dropped": m["source"]["late_dropped"]}, indent=2))


if __name__ == "__main__":
    main()
