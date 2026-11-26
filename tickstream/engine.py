"""Process topology and orchestration.

    Source process ──(bounded queue per worker)──► Worker 0..N-1 ──(results queue)──► Sink (main)

* Source: tickstream.source.Source; sends every worker one message per CSV block
  (its partition's events + the current watermark), so all workers advance event time
  together even if their partition is empty in that block.
* Worker: tickstream.operator.WindowEmaOperator on its partition (keyed state).
* Sink (main process): collects emitted window results and advisories, writes the live
  advisory feed, measures latency, samples CPU utilization, writes the run outputs.
"""
import json
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

from .config import WINDOW_MS, RunConfig

END = "end"


# --------------------------------------------------------------------------- helpers
def pin(cores):
    if cores:
        psutil.Process().cpu_affinity(list(cores))


def load_weights(path):
    import csv
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        return {row[0]: float(row[1]) for row in r}


def load_subscriptions(path):
    if not path or not Path(path).exists():
        if path:
            print(f"warning: subscription file {path} not found; no symbols subscribed")
        return set()
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def pct(a, qs=(50, 90, 99, 99.9)):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return {}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"max": float(a.max()),
                                                               "mean": float(a.mean())}


# --------------------------------------------------------------------------- source process
def source_main(cfg: RunConfig, worker_queues, results_q):
    pin(cfg.cores)
    import pyarrow as pa
    from .source import Partitioner, Source
    threads = cfg.source_threads or (len(cfg.cores) if cfg.cores else None)
    if threads:
        pa.set_cpu_count(threads)
    weights = load_weights(cfg.weights_file) if cfg.partitioning == "balanced" else None
    part = Partitioner(cfg.workers, cfg.partitioning, weights)
    src = Source(cfg.files, cfg.workers, part, cfg.lateness_ms, cfg.block_size, cfg.max_rows,
                 use_threads=(threads or 2) > 1)
    subs = load_subscriptions(cfg.subscriptions_file)
    batches = src.batches()
    if cfg.preload:
        # Processing-only mode: parse everything first, then stream from memory, so the
        # measurement isolates routing + keyed window processing from CSV parsing.
        batches = list(batches)
    t_start = time.time()
    results_q.put(("stream_start", t_start))
    seq = 0
    for t_read, parts, new_syms, wm in batches:
        if cfg.preload:
            t_read = time.time()  # latency measured from the moment the batch is emitted
        if new_syms:
            names = [src.registry.names[s] for s in new_syms]
            results_q.put(("symbols", new_syms, names, [src.registry.sectypes[s] for s in new_syms]))
            sub_new = [s for s, n in zip(new_syms, names) if n in subs]
        else:
            sub_new = []
        for w, q in enumerate(worker_queues):
            my_subs = [s for s in sub_new if part.of_sym[s] == w]
            q.put(("batch", seq, t_read, wm, parts[w], my_subs))
        seq += 1
    t_end = time.time()
    for q in worker_queues:
        q.put((END, seq, t_end))
    stats = dict(src.stats, wall_s=t_end - t_start, symbols=len(src.registry.names))
    results_q.put(("source_stats", stats, part.of_sym[: len(src.registry.names)].tolist()))


# --------------------------------------------------------------------------- worker process
def worker_main(wid: int, cfg: RunConfig, in_q, results_q):
    pin(cfg.cores)
    from .operator import WindowEmaOperator
    op = WindowEmaOperator(warmup_windows=cfg.warmup_windows)
    events = 0
    busy = 0.0
    wait = 0.0
    batch_latency = []  # wall time from source starting to read a block -> worker done with it
    close_latency = []
    evaluated = 0
    while True:
        t0 = time.time()
        msg = in_q.get()
        t1 = time.time()
        wait += t1 - t0
        if msg[0] == END:
            res = op.process([], [], [], np.inf, cfg.emit_all)  # flush all open windows
            if len(res["sym"]):
                results_q.put(("res", wid, msg[2], time.time(), res))
            break
        _, seq, t_read, wm, part, subs = msg
        if subs:
            op.subscribe(subs)
        if part is None:
            res = op.process([], [], [], wm, cfg.emit_all)
        else:
            sym, t, price = part
            events += len(sym)
            res = op.process(sym, t, price, wm, cfg.emit_all)
        t2 = time.time()
        busy += t2 - t1
        batch_latency.append(t2 - t_read)
        if op.last_evaluated:
            # window-evaluation latency: source began reading the block whose watermark closed
            # the window(s) -> EMAs/crossovers computed (independent of what is emitted)
            close_latency.append(t2 - t_read)
            evaluated += op.last_evaluated
        if len(res["sym"]):
            results_q.put(("res", wid, t_read, t2, res))
    results_q.put(("worker_stats", wid, dict(events=events, busy_s=busy, wait_s=wait,
                                             late_dropped_in_operator=op.late_dropped,
                                             windows_evaluated=evaluated,
                                             batch_latency_s=pct(batch_latency),
                                             close_latency_raw=close_latency)))


# --------------------------------------------------------------------------- CPU sampling
class CpuSampler(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []  # (t, process-tree cpu %, system cpu %)
        self.stop_evt = threading.Event()
        self.root = psutil.Process()

    def run(self):
        procs = {}
        psutil.cpu_percent(None)
        while not self.stop_evt.is_set():
            tree = [self.root] + self.root.children(recursive=True)
            total = 0.0
            for p in tree:
                try:
                    if p.pid not in procs:
                        procs[p.pid] = p
                        p.cpu_percent(None)
                        continue
                    total += procs[p.pid].cpu_percent(None)
                except psutil.Error:
                    pass
            self.samples.append((time.time(), total, psutil.cpu_percent(None)))
            self.stop_evt.wait(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join()


# --------------------------------------------------------------------------- orchestration
def run(cfg: RunConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pin(cfg.cores)
    ctx = mp.get_context("spawn")
    worker_queues = [ctx.Queue(maxsize=cfg.queue_size) for _ in range(cfg.workers)]
    results_q = ctx.Queue()

    sampler = CpuSampler()
    sampler.start()
    t_start = time.time()
    procs = [ctx.Process(target=worker_main, args=(w, cfg, worker_queues[w], results_q), daemon=True)
             for w in range(cfg.workers)]
    procs.append(ctx.Process(target=source_main, args=(cfg, worker_queues, results_q), daemon=True))
    for p in procs:
        p.start()

    names, sectypes = {}, {}
    chunks = []
    result_latency = []
    worker_stats, source_stats, of_sym = {}, None, None
    live = open(out / "advisories_live.jsonl", "w") if cfg.live_advisories else None
    t_first_result = None
    t_stream_start = None
    while len(worker_stats) < cfg.workers or source_stats is None:
        try:
            msg = results_q.get(timeout=2.0)
        except queue.Empty:
            dead = [p for p in procs if p.exitcode not in (None, 0)]
            if dead:
                for p in procs:
                    p.terminate()
                raise RuntimeError(f"child process failed: {[(p.name, p.exitcode) for p in dead]}")
            continue
        kind = msg[0]
        if kind == "symbols":
            for s, n, st in zip(msg[1], msg[2], msg[3]):
                names[s], sectypes[s] = n, st
        elif kind == "res":
            _, wid, t_read, t_done, res = msg
            now = time.time()
            t_first_result = t_first_result or now
            result_latency.append(now - t_read)
            chunks.append(res)
            if live is not None:
                adv = np.flatnonzero(res["advice"] != 0)
                for i in adv:
                    s = int(res["sym"][i])
                    live.write(json.dumps(dict(
                        symbol=names.get(s, str(s)), window_end=_fmt_ts((int(res["win"][i]) + 1) * WINDOW_MS),
                        advice="BUY" if res["advice"][i] > 0 else "SELL", close=float(res["close"][i]),
                        ema38=float(res["ema38"][i]), ema100=float(res["ema100"][i]),
                        latency_ms=round((now - t_read) * 1000, 2))) + "\n")
                if len(adv):
                    live.flush()
        elif kind == "stream_start":
            t_stream_start = msg[1]
        elif kind == "worker_stats":
            worker_stats[msg[1]] = msg[2]
        elif kind == "source_stats":
            source_stats, of_sym = msg[1], msg[2]
    t_end = time.time()
    sampler.stop()
    for p in procs:
        p.join()
    if live is not None:
        live.close()

    # Throughput is measured from the moment the source starts streaming (excludes process
    # start-up, and in preload mode the up-front parsing) until the sink has everything.
    wall = t_end - t_stream_start
    _write_results(out, chunks, names, sectypes)
    cpu = np.array([(s[1], s[2]) for s in sampler.samples]) if sampler.samples else np.zeros((0, 2))
    ev = [worker_stats[w]["events"] for w in range(cfg.workers)]
    close_lat = np.concatenate([np.asarray(worker_stats[w].pop("close_latency_raw"), dtype=float)
                                for w in range(cfg.workers)])
    metrics = dict(
        config=asdict(cfg),
        wall_s=wall,
        total_wall_s=t_end - t_start,  # incl. process start-up (and preload parsing)
        rows=source_stats["rows"],
        price_events=source_stats["price_events"],
        throughput_rows_per_s=source_stats["rows"] / wall,
        throughput_price_events_per_s=source_stats["price_events"] / wall,
        source=source_stats,
        workers=worker_stats,
        worker_event_share=[e / max(1, sum(ev)) for e in ev],
        load_imbalance=(max(ev) / (sum(ev) / len(ev))) if sum(ev) else None,  # max/mean
        window_close_latency_s=pct(close_lat),   # all symbols, at the worker
        result_latency_s=pct(result_latency),    # emitted results, at the sink
        cpu_process_tree_pct=pct(cpu[:, 0]) if len(cpu) else {},
        cpu_system_pct=pct(cpu[:, 1]) if len(cpu) else {},
        n_cores=len(cfg.cores) if cfg.cores else psutil.cpu_count(),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "cpu_timeline.csv", "w") as f:
        f.write("t,process_tree_pct,system_pct\n")
        for t, a, b in sampler.samples:
            f.write(f"{t - t_start:.2f},{a:.1f},{b:.1f}\n")
    return metrics


def _fmt_ts(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000))


def _write_results(out: Path, chunks, names, sectypes):
    import polars as pl
    if chunks:
        cols = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    else:
        cols = {k: np.empty(0) for k in ("sym", "win", "close", "ema38", "ema100", "prev38", "prev100", "advice")}
    df = pl.DataFrame(cols).with_columns(
        symbol=pl.Series([names.get(int(s), str(s)) for s in cols["sym"]], dtype=pl.Utf8),
        sectype=pl.Series([sectypes.get(int(s)) for s in cols["sym"]], dtype=pl.Utf8),
        window_start=pl.from_epoch(pl.Series(cols["win"].astype(np.int64) * WINDOW_MS), time_unit="ms"),
    ).sort(["symbol", "win"])
    df.write_parquet(out / "windows.parquet")
    df.filter(pl.col("advice") != 0).write_parquet(out / "advisories.parquet")
