"""Benchmark harness: runs the engine over an experiment matrix and collects a summary CSV.

Experiments (select with --exp, several allowed):
  cores      end-to-end scaling: cores in --cores-list, hash partitioning
  preload    processing-only scaling (input parsed up front), cores in --cores-list
  partition  hash vs balanced partitioning at each core count (preload, isolates the workers)
  lateness   allowed lateness in --lateness-list: late drops vs result latency
  block      source block size in --block-list (MB): throughput vs latency
Each configuration is repeated --reps times.

Usage: python scripts/bench.py FILES... --exp cores preload --reps 2 --tag tue
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def run_one(files, out, args):
    cmd = [PY, "-m", "tickstream", *files, "--out", str(out), *args]
    t = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    m = json.loads((out / "metrics.json").read_text())
    w = m["workers"]
    return dict(
        wall_s=round(m["wall_s"], 3),
        total_wall_s=round(m["total_wall_s"], 3),
        rows=m["rows"], price_events=m["price_events"],
        rows_per_s=round(m["throughput_rows_per_s"]),
        price_events_per_s=round(m["throughput_price_events_per_s"]),
        load_imbalance=round(m["load_imbalance"] or 0, 4),
        worker_busy_max_s=round(max(v["busy_s"] for v in w.values()), 3),
        worker_busy_mean_s=round(sum(v["busy_s"] for v in w.values()) / len(w), 3),
        source_read_s=round(m["source"]["read_s"], 3),
        source_transform_s=round(m["source"]["transform_s"], 3),
        late_dropped=m["source"]["late_dropped"],
        lat_p50_ms=round(m["window_close_latency_s"].get("p50", 0) * 1000, 2),
        lat_p99_ms=round(m["window_close_latency_s"].get("p99", 0) * 1000, 2),
        lat_max_ms=round(m["window_close_latency_s"].get("max", 0) * 1000, 2),
        sink_lat_p50_ms=round(m["result_latency_s"].get("p50", 0) * 1000, 2),
        cpu_mean_pct=round(m["cpu_process_tree_pct"].get("mean", 0), 1),
        cpu_p90_pct=round(m["cpu_process_tree_pct"].get("p90", 0), 1),
        harness_s=round(time.time() - t, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exp", nargs="+", default=["cores"])
    ap.add_argument("--cores-list", type=int, nargs="+", default=[1, 2, 4, 6, 8, 14])
    ap.add_argument("--lateness-list", type=int, nargs="+", default=[0, 10, 100, 500, 1000])
    ap.add_argument("--block-list", type=float, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--fixed-cores", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--warmup-run", action="store_true",
                    help="one discarded run first (warms the OS file cache)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra engine args")
    a = ap.parse_args()
    files = [g for f in a.files for g in (sorted(glob.glob(f)) or [f])]

    configs = []  # (experiment, label dict, engine args)
    if "cores" in a.exp:
        configs += [("cores", dict(cores=c), ["--cores", str(c)]) for c in a.cores_list]
    if "preload" in a.exp:
        configs += [("preload", dict(cores=c), ["--cores", str(c), "--preload"]) for c in a.cores_list]
    if "partition" in a.exp:
        configs += [("partition", dict(cores=c, partitioning=p),
                     ["--cores", str(c), "--preload", "--partitioning", p])
                    for c in a.cores_list if c > 1 for p in ("hash", "balanced")]
    if "lateness" in a.exp:
        configs += [("lateness", dict(cores=a.fixed_cores, lateness_ms=l),
                     ["--cores", str(a.fixed_cores), "--lateness-ms", str(l)]) for l in a.lateness_list]
    if "block" in a.exp:
        configs += [("block", dict(cores=a.fixed_cores, block_mb=b),
                     ["--cores", str(a.fixed_cores), "--block-mb", str(b)]) for b in a.block_list]

    root = Path("results/bench") / a.tag
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.csv"
    write_header = not summary.exists()
    if a.warmup_run:
        print("warm-up run (discarded) ...", flush=True)
        run_one(files, Path("results/runs") / a.tag / "warmup", ["--cores", str(a.fixed_cores)] + a.extra)
    with open(summary, "a", newline="") as f:
        wr = None
        for exp, label, args in configs:
            for rep in range(a.reps):
                name = f"{exp}_" + "_".join(f"{k}{v}" for k, v in label.items()) + f"_r{rep}"
                out = Path("results/runs") / a.tag / name
                print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
                res = run_one(files, out, args + a.extra)
                row = dict(experiment=exp, rep=rep, partitioning=label.get("partitioning", "hash"),
                           cores=label.get("cores"), lateness_ms=label.get("lateness_ms", 1000),
                           block_mb=label.get("block_mb", 16)) | res
                print("   ", {k: row[k] for k in ("wall_s", "rows_per_s", "price_events_per_s",
                                                  "load_imbalance", "lat_p50_ms", "late_dropped")},
                      flush=True)
                if wr is None:
                    wr = csv.DictWriter(f, fieldnames=list(row))
                    if write_header:
                        wr.writeheader()
                wr.writerow(row)
                f.flush()
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
