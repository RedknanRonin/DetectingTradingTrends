"""Live market-data feed: replays DEBS 2022 CSV day files over TCP at their real arrival rate.

Each event is sent when the replay clock reaches its arrival time (system Date + Time columns),
so the stream processor sees the data as it would have arrived from the three exchanges
(FR, NL, ETR interleaved, as in the files). Protocol: plain text over TCP, the CSV header line
first, then one CSV line per event; the server closes the connection at the end.

    python scripts/feed_server.py C:/data/debs2022/*10-11-21.csv --port 9000 --speed 1
    python -m tickstream --socket 127.0.0.1:9000 --clock-rate 1 --out results/runs/live

Options:
  --speed S        replay S x faster than real time (clock_rate of the processor should match)
  --fast-until T   send everything before arrival time T (HH:MM or "DD-MM-YYYY HH:MM") without
                   pacing, e.g. to build up EMA history quickly, then switch to real time
  --until T        stop (close the connection) at arrival time T
  --max-gap-s G    cap the wait between two events (e.g. skip the night between day files)
"""
import argparse
import bisect
import collections
import datetime as dt
import glob
import socket
import sys
import time

HEADER_SKIP_ROWS = 11  # comment lines before the header, one '#' description line after it
_EPOCH = dt.datetime(1970, 1, 1)


def day_ms(date: str, cache={}) -> int:
    v = cache.get(date)
    if v is None:
        v = cache[date] = int((dt.datetime.strptime(date, "%d-%m-%Y") - _EPOCH).total_seconds() * 1000)
    return v


def arrival_ms(line: bytes, i_date, i_time):
    """Date 'DD-MM-YYYY' + Time 'HH:MM:SS.fff' of a CSV line -> epoch ms, or None if malformed."""
    try:
        fields = line.split(b",", i_time + 1)
        d, t = fields[i_date].decode(), fields[i_time].decode()
        return day_ms(d) + ((int(t[0:2]) * 60 + int(t[3:5])) * 60 + int(t[6:8])) * 1000 + int(t[9:12] or 0)
    except (ValueError, IndexError):
        return None


def parse_time_arg(s, first_day):
    """'HH:MM' (on the first file's day) or 'DD-MM-YYYY HH:MM' -> epoch ms."""
    if s is None:
        return None
    if " " in s:
        d, hm = s.split(" ", 1)
    else:
        d, hm = first_day, s
    h, m = hm.split(":")[:2]
    return day_ms(d) + (int(h) * 60 + int(m)) * 60_000


def first_day_of(path):
    name = path.rsplit("-day-", 1)[-1].split(".")[0]  # debs2022-gc-trading-day-10-11-21.csv
    dd, mm, yy = name.split("-")
    return f"{dd}-{mm}-20{yy}"


def lines_of(files):
    """Yields the header once, then the data lines of all files in order (bytes, with '\\n')."""
    header = None
    for path in files:
        with open(path, "rb") as f:
            for _ in range(HEADER_SKIP_ROWS):
                f.readline()
            h = f.readline()
            if header is None:
                header = h
                yield h
            for line in f:
                if line.startswith(b"#") or not line.strip():
                    continue
                yield line if line.endswith(b"\n") else line + b"\n"


class ReplayClock:
    """Robust 'current arrival time' of the feed, used only for pacing.

    File order is the true arrival order, but some rows carry bogus Date+Time stamps:
    00:00:00 placeholders, the opening snapshot (incl. a block of several hundred rows stamped
    with the previous evening's ~17:35 last-update time) and scattered far-future values.
    Measured on Mon/Wed/Fri, far-ahead rows come in runs of at most ~540 rows, so the clock is
    the running max of the median of the last 4001 stamps (sampled: every 8th row, window 501
    samples, same ratio to the outlier runs - keeps the feeder fast): it is never more than
    ~5 min ahead (only briefly, at the file start) and trails the true time by ~1 s (p99 15 s,
    in sparse night hours) - i.e. the feed runs ~1 s late, consistently.
    """

    SAMPLE = 8  # read the stamp of every 8th row only

    def __init__(self, window=501):
        self.window = window
        self.recent = collections.deque()
        self.sorted = []
        self.now = None

    def update(self, a):
        if a is not None:
            self.recent.append(a)
            bisect.insort(self.sorted, a)
            if len(self.recent) > self.window:
                old = self.recent.popleft()
                del self.sorted[bisect.bisect_left(self.sorted, old)]
            if len(self.recent) >= self.window // 2:
                med = self.sorted[len(self.sorted) // 2]
                if self.now is None or med > self.now:
                    self.now = med
        return self.now


def serve(conn, files, speed, fast_until, until, max_gap_s, status_s=10.0):
    it = lines_of(files)
    header = next(it)
    cols = [c.strip() for c in header.decode().split(",")]
    i_date, i_time = cols.index("Date"), cols.index("Time")
    conn.sendall(header if header.endswith(b"\n") else header + b"\n")

    clock = ReplayClock()
    buf, buf_bytes = [], 0
    sent = sent_at_status = 0
    wall0 = arr0 = None          # anchor: wall time <-> feed clock (arrival ms)
    prev = None                  # feed clock at the previous row
    last_status = time.time()

    def flush():
        nonlocal buf, buf_bytes
        if buf:
            conn.sendall(b"".join(buf))
            buf, buf_bytes = [], 0

    every = ReplayClock.SAMPLE
    for line in it:
        if sent % every == 0:
            c = clock.update(arrival_ms(line, i_date, i_time))
        else:
            c = prev
        if c is not None and c != prev:
            if until is not None and c >= until:
                break
            if fast_until is None or c >= fast_until:   # real-time phase: pace on the clock
                now = time.time()
                if wall0 is None:
                    wall0, arr0 = now, c
                elif max_gap_s is not None and prev is not None and (c - prev) / 1000 / speed > max_gap_s:
                    arr0 += (c - prev) - int(max_gap_s * 1000 * speed)  # shorten long gaps
                target = wall0 + (c - arr0) / 1000 / speed
                if target > now:
                    flush()  # everything due so far goes out before we wait
                    time.sleep(target - now)
            prev = c
        buf.append(line)
        buf_bytes += len(line)
        sent += 1
        if buf_bytes >= 1 << 16:
            flush()
        if sent % 1024 == 0:
            now = time.time()
            if now - last_status >= status_s:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime((c or 0) / 1000))
                print(f"[feed] sent={sent:,}  ({(sent - sent_at_status) / (now - last_status):,.0f} lines/s)"
                      f"  feed clock={ts}", flush=True)
                last_status, sent_at_status = now, sent
    flush()
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fast-until", default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--max-gap-s", type=float, default=None)
    a = ap.parse_args()
    files = [g for f in a.files for g in (sorted(glob.glob(f)) or [f])]
    first_day = first_day_of(files[0])
    fast_until = parse_time_arg(a.fast_until, first_day)
    until = parse_time_arg(a.until, first_day)

    with socket.create_server((a.host, a.port)) as srv:
        print(f"[feed] listening on {a.host}:{a.port}, {len(files)} file(s), speed {a.speed}x", flush=True)
        conn, peer = srv.accept()
        print(f"[feed] client {peer[0]}:{peer[1]} connected", flush=True)
        t0 = time.time()
        with conn:
            try:
                n = serve(conn, files, a.speed, fast_until, until, a.max_gap_s)
            except (BrokenPipeError, ConnectionResetError):
                print("[feed] client disconnected", flush=True)
                sys.exit(1)
            conn.shutdown(socket.SHUT_WR)
        print(f"[feed] done: {n:,} lines in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
