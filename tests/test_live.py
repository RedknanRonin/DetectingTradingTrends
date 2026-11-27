"""Live mode: socket source (time-based batching, idle watermark) and the feeder's replay clock."""
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from tickstream.config import WINDOW_MS
from tickstream.source import Partitioner, SocketSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feed_server import ReplayClock  # noqa: E402

HEADER = b"ID,SecType,Date,Time,Last,Trading time,Trading date\n"


def line(sym, hms, price):
    # empty Trading date, then the trailing comma every data row has (n_header + 1 fields)
    return f"{sym},E,10-11-2021,{hms},{price},{hms},,\n".encode()


def serve_once(chunks):
    """TCP server on a free port; sends (bytes | pause seconds) items, then closes."""
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        with conn:
            for c in chunks:
                if isinstance(c, (int, float)):
                    time.sleep(c)
                else:
                    conn.sendall(c)
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return f"127.0.0.1:{port}"


def collect(src):
    out = []
    for t0, parts, new, wm in src.batches():
        n = sum(len(p[0]) for p in parts if p is not None)
        out.append((time.time(), n, wm))
    return out


def test_socket_source_batches_by_time_and_matches_events():
    addr = serve_once([HEADER, line("A.FR", "09:00:01.000", 10.0), 0.3,
                       line("A.FR", "09:00:02.000", 11.0) + line("B.FR", "09:00:03.000", 5.0)])
    src = SocketSource(addr, 2, Partitioner(2), lateness_ms=1000, block_size=1 << 20,
                       flush_ms=50, idle_ms=10_000, status_s=0)
    got = collect(src)
    assert sum(n for _, n, _ in got) == 3
    assert len([g for g in got if g[1]]) >= 2      # first line was flushed before the pause
    assert src.stats["price_events"] == 3 and src.stats["late_dropped"] == 0


def test_idle_watermark_closes_window_when_feed_is_quiet():
    # one event at 09:04:59, then silence. With clock_rate 1000, 0.2 s idle = 200 s of event
    # time, so the watermark must pass the window end (09:05) while no data arrives.
    addr = serve_once([HEADER, line("A.FR", "09:04:59.000", 10.0), 1.0])
    src = SocketSource(addr, 1, Partitioner(1), lateness_ms=1000, block_size=1 << 20,
                       flush_ms=20, idle_ms=200, clock_rate=1000, status_s=0)
    got = collect(src)
    day = 1636502400000  # 10-11-2021 00:00 UTC
    window_end = day + 9 * 3600_000 + 5 * 60_000
    assert window_end % WINDOW_MS == 0
    assert src.stats["idle_advances"] >= 1
    assert max(wm for _, _, wm in got if wm is not None) >= window_end


def test_replay_clock_ignores_outlier_runs():
    clock = ReplayClock(window=101)
    t = np.arange(2000) * 100 + 9 * 3600_000          # ordered stamps, 100 ms apart
    stamps = t.tolist()
    stamps[500:540] = [17 * 3600_000] * 40            # run of far-future stamps (< window/2)
    stamps[900:1000:3] = [0] * 34                     # scattered 00:00:00 placeholders
    seen = [clock.update(a) for a in stamps]
    valid = [c for c in seen if c is not None]
    assert max(valid) <= t[-1]                         # never jumped to 17:00
    assert t[-1] - valid[-1] < 60 * 100                # trails the true time by ~window/2 rows
    assert all(b >= a for a, b in zip(valid, valid[1:]))  # monotone
