"""Source stage: read CSV day files in order, normalize, filter, mark late events, route.

Per CSV block (D10):
  * parse only the needed columns with pyarrow (columnar, C++),
  * keep price events only (Last > 0 and Trading time present); others are counted,
  * event time = coalesce(Trading date, Date) + Trading time  (D4, see profiling),
  * map symbol strings to global integer ids (symbol registry, incl. SecType metadata),
  * drop late events against the running watermark, row by row (D5),
  * split the block by partition (D9) preserving per-symbol order.
"""
import datetime as dt
import select
import socket
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .config import WINDOW_MS
from .schema import (COL_DATE, COL_ID, COL_LAST, COL_SECTYPE, COL_TIME, COL_TRADING_DATE,
                     COL_TRADING_TIME)

HEADER_SKIP_ROWS = 11  # comment lines before the header; one '#...' description line after it
SOURCE_COLUMNS = [COL_ID, COL_SECTYPE, COL_DATE, COL_TIME, COL_LAST, COL_TRADING_TIME, COL_TRADING_DATE]
STALE_MS_DEFAULT = 5 * 60 * 1000  # Trading time this far behind arrival time = invalid stamp
_EPOCH = dt.datetime(1970, 1, 1)
_NO_TIME = -(2**62)  # "no event seen yet" (safe from overflow when subtracting lateness)


def read_header(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for _ in range(HEADER_SKIP_ROWS):
            f.readline()
        return [c.strip() for c in f.readline().rstrip("\r\n").split(",")]


class InvalidRowCounter:
    """pyarrow invalid-row handler: skip malformed rows (e.g. a truncated last line), count them."""

    def __init__(self):
        self.count = 0
        self.examples = []

    def __call__(self, row):
        self.count += 1
        if len(self.examples) < 5:
            self.examples.append(row.text[:120])
        return "skip"


def open_csv(path: str, block_size: int, use_threads: bool = True, invalid_rows=None):
    # Data rows end with a trailing comma (40 fields vs 39 header names): add a dummy column.
    names = read_header(path) + ["_trailing"]
    return pacsv.open_csv(
        path,
        parse_options=pacsv.ParseOptions(invalid_row_handler=invalid_rows),
        read_options=pacsv.ReadOptions(skip_rows=HEADER_SKIP_ROWS + 2, column_names=names,
                                       block_size=block_size, use_threads=use_threads),
        convert_options=convert_options(),
    )


def convert_options():
    return pacsv.ConvertOptions(
        include_columns=SOURCE_COLUMNS,
        strings_can_be_null=True,  # empty field -> null (needed for coalesce / validity)
        column_types={c: pa.string() for c in SOURCE_COLUMNS if c != COL_LAST}
        | {COL_LAST: pa.float64()},
    )


def _day_ms(date_str: str) -> int:
    d = dt.datetime.strptime(date_str, "%d-%m-%Y")
    return int((d - _EPOCH).total_seconds() * 1000)


def time_of_day_ms(tt: pa.Array) -> np.ndarray:
    """'HH:MM:SS.fff[f]' -> milliseconds since midnight (vectorized)."""
    def part(a, b):
        return pc.cast(pc.utf8_slice_codeunits(tt, a, b), pa.int64()).to_numpy(zero_copy_only=False)
    h, m, s = part(0, 2), part(3, 5), part(6, 8)
    frac = pc.utf8_slice_codeunits(tt, 9, 12)  # first 3 fractional digits = ms
    frac = pc.utf8_rpad(frac, 3, "0")
    ms = pc.cast(frac, pa.int64()).to_numpy(zero_copy_only=False)
    return ((h * 60 + m) * 60 + s) * 1000 + ms


class SymbolRegistry:
    """Global symbol -> integer id map (+ SecType), owned by the source."""

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.names: list[str] = []
        self.sectypes: list[str] = []

    def lookup(self, names: list[str], sectypes: list[str]) -> tuple[np.ndarray, list[int]]:
        """Ids for a list of (unique) names; returns ids and the list of newly added ids."""
        out = np.empty(len(names), dtype=np.int64)
        new = []
        for i, (n, st) in enumerate(zip(names, sectypes)):
            sid = self.ids.get(n)
            if sid is None:
                sid = len(self.names)
                self.ids[n] = sid
                self.names.append(n)
                self.sectypes.append(st)
                new.append(sid)
            out[i] = sid
        return out, new


class Partitioner:
    """Symbol id -> worker index. 'hash' = crc32(name) mod N; 'balanced' = greedy LPT map."""

    def __init__(self, n_workers: int, mode: str = "hash", weights: dict[str, float] | None = None):
        self.n = n_workers
        self.mode = mode
        self.fixed: dict[str, int] = {}
        if mode == "balanced":
            if not weights:
                raise ValueError("balanced partitioning needs symbol weights")
            self.fixed = balanced_assignment(weights, n_workers)
        elif mode != "hash":
            raise ValueError(f"unknown partitioning {mode}")
        self.of_sym = np.zeros(0, dtype=np.int64)  # indexed by global symbol id

    def assign(self, sym_ids: list[int], names: list[str]) -> None:
        if sym_ids:
            need = max(sym_ids) + 1
            if need > len(self.of_sym):
                grown = np.zeros(max(need, 2 * len(self.of_sym)), dtype=np.int64)
                grown[: len(self.of_sym)] = self.of_sym
                self.of_sym = grown
        for sid, name in zip(sym_ids, names):
            p = self.fixed.get(name)
            self.of_sym[sid] = p if p is not None else zlib.crc32(name.encode()) % self.n


def balanced_assignment(weights: dict[str, float], n: int) -> dict[str, int]:
    """Longest-processing-time-first greedy: heaviest symbol -> least loaded worker."""
    load = np.zeros(n)
    out = {}
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        p = int(np.argmin(load))
        out[name] = p
        load[p] += w
    return out


class Source:
    def __init__(self, files, n_workers, partitioner: Partitioner, lateness_ms: int,
                 block_size: int, max_rows=None, use_threads=True, stale_ms=STALE_MS_DEFAULT):
        self.files = files
        self.n_workers = n_workers
        self.part = partitioner
        self.lateness = lateness_ms
        self.block_size = block_size
        self.max_rows = max_rows
        self.use_threads = use_threads
        self.stale_ms = stale_ms
        self.registry = SymbolRegistry()
        self.max_t = _NO_TIME  # max event time seen so far
        self.day_cache: dict[str, int] = {}
        self.invalid_rows = InvalidRowCounter()
        self.stats = dict(rows=0, price_events=0, non_price_events=0, late_dropped=0,
                          stale_timestamp_dropped=0,
                          invalid_rows_skipped=0, batches=0, read_s=0.0, transform_s=0.0)

    @property
    def watermark(self) -> int | None:
        if self.max_t == _NO_TIME:
            return None
        return int(self.max_t - self.lateness)

    def batches(self):
        """Yields (t_read_start, per_partition_arrays, new_symbols, watermark)."""
        for path in self.files:
            reader = open_csv(path, self.block_size, self.use_threads, self.invalid_rows)
            while True:
                t0 = time.time()
                try:
                    rb = reader.read_next_batch()
                except StopIteration:
                    break
                if self.max_rows and self.stats["rows"] + rb.num_rows > self.max_rows:
                    rb = rb.slice(0, self.max_rows - self.stats["rows"])  # exact row limit
                t1 = time.time()
                parts, new_syms = self.transform(rb)
                t2 = time.time()
                self.stats["read_s"] += t1 - t0
                self.stats["transform_s"] += t2 - t1
                self.stats["batches"] += 1
                self.stats["invalid_rows_skipped"] = self.invalid_rows.count
                yield t0, parts, new_syms, self.watermark
                if self.max_rows and self.stats["rows"] >= self.max_rows:
                    return

    def _days(self, dates: pa.Array) -> np.ndarray:
        """'DD-MM-YYYY' strings -> epoch ms of that midnight (few distinct values per block)."""
        enc = pc.dictionary_encode(dates)
        uniq = enc.dictionary.to_pylist()
        for d in uniq:
            if d not in self.day_cache:
                self.day_cache[d] = _day_ms(d)
        day = np.array([self.day_cache[d] for d in uniq], dtype=np.int64)
        return day[enc.indices.to_numpy(zero_copy_only=False)]

    def transform(self, rb: pa.RecordBatch):
        n_rows = rb.num_rows
        self.stats["rows"] += n_rows
        last = rb.column(COL_LAST)
        tt = rb.column(COL_TRADING_TIME)
        is_price = pc.and_(pc.fill_null(pc.greater(last, 0.0), False), pc.is_valid(tt))
        rb = rb.filter(is_price)
        n = rb.num_rows
        self.stats["price_events"] += n
        self.stats["non_price_events"] += n_rows - n
        if n == 0:
            return [None] * self.n_workers, []

        # event time = coalesce(Trading date, Date) + Trading time
        t = (self._days(pc.coalesce(rb.column(COL_TRADING_DATE), rb.column(COL_DATE)))
             + time_of_day_ms(rb.column(COL_TRADING_TIME)))

        # Data-quality rule (I14): a Trading time far BEHIND the arrival time (system Date + Time)
        # is an invalid stamp (e.g. the Wed 18:56 burst of index events stamped 01:00:00.000),
        # not a late event. Dropped and counted separately, before it can touch the watermark.
        if self.stale_ms is not None:
            arrival = self._days(rb.column(COL_DATE)) + time_of_day_ms(rb.column(COL_TIME))
            stale = t < arrival - self.stale_ms
            if stale.any():
                self.stats["stale_timestamp_dropped"] += int(stale.sum())
                keep = ~stale
                rb = rb.filter(pa.array(keep))
                t = t[keep]
                if rb.num_rows == 0:
                    return [None] * self.n_workers, []

        # symbols -> global ids (only the batch's distinct symbols touch Python)
        ienc = pc.dictionary_encode(rb.column(COL_ID))
        names = ienc.dictionary.to_pylist()
        idx = ienc.indices.to_numpy(zero_copy_only=False)
        # SecType of the first occurrence of each distinct symbol
        first = np.unique(idx, return_index=True)[1]
        st_all = rb.column(COL_SECTYPE).to_numpy(zero_copy_only=False)
        sectypes = [None] * len(names)
        for k, pos in zip(np.unique(idx), first):
            sectypes[k] = st_all[pos]
        ids, new = self.registry.lookup(names, sectypes)
        self.part.assign(new, [self.registry.names[s] for s in new])
        sym = ids[idx]
        price = rb.column(COL_LAST).to_numpy(zero_copy_only=False)

        # late filtering against the watermark *before* each row (event-at-a-time semantics):
        # row is late iff its window was already closed, i.e. win(t) < floor(wm_prev / W).
        running = np.maximum.accumulate(np.concatenate(([self.max_t], t)))
        wm_prev = running[:-1] - self.lateness
        late = (t // WINDOW_MS) < np.floor_divide(wm_prev, WINDOW_MS)
        self.max_t = int(running[-1])
        if late.any():
            self.stats["late_dropped"] += int(late.sum())
            keep = ~late
            sym, t, price = sym[keep], t[keep], price[keep]

        # route: stable sort by partition keeps per-symbol arrival order
        pid = self.part.of_sym[sym]
        order = np.argsort(pid, kind="stable")
        counts = np.bincount(pid, minlength=self.n_workers)
        bounds = np.cumsum(counts)[:-1]
        parts = []
        for s, tt_, p in zip(np.split(sym[order], bounds), np.split(t[order], bounds),
                             np.split(price[order], bounds)):
            parts.append((s, tt_, p) if len(s) else None)
        return parts, new


class SocketSource(Source):
    """Live variant of the source: reads CSV lines from a TCP socket (scripts/feed_server.py).

    Same transform (filters, event time, late filtering, routing) as the file source; only
    batching differs, because a live feed is slow compared with a file:
      * a batch is cut every `flush_ms` (time-based) or when `block_size` bytes are buffered,
        whichever comes first, so a quiet feed does not wait for a full block;
      * idle watermark: if no data arrives for `idle_ms`, event time is advanced with the wall
        clock (x `clock_rate`, the feeder's replay speed) so open windows still close when the
        whole market goes quiet. Without data the watermark would otherwise never move.
    The batch timestamp passed downstream is the wall time its first byte arrived, so the
    latency metrics include the buffering delay of up to `flush_ms`.
    """

    def __init__(self, address: str, n_workers, partitioner, lateness_ms, block_size,
                 flush_ms=100, idle_ms=2000, clock_rate=1.0, stale_ms=STALE_MS_DEFAULT,
                 status_s=5.0):
        super().__init__([], n_workers, partitioner, lateness_ms, block_size, stale_ms=stale_ms)
        host, port = address.rsplit(":", 1)
        self.address = (host or "127.0.0.1", int(port))
        self.flush_s = flush_ms / 1000
        self.idle_s = idle_ms / 1000
        self.clock_rate = clock_rate
        self.status_s = status_s
        self.stats |= dict(idle_advances=0, bytes=0)
        self._clock_wall = None  # wall time at which max_t was last set by real data

    def _connect(self, timeout_s=30.0):
        t_end = time.time() + timeout_s
        while True:
            try:
                return socket.create_connection(self.address, timeout=5)
            except OSError:
                if time.time() > t_end:
                    raise
                time.sleep(0.5)

    def _parse(self, chunk: bytes):
        return pacsv.read_csv(
            pa.BufferReader(chunk),
            read_options=pacsv.ReadOptions(column_names=self.names, use_threads=False),
            parse_options=pacsv.ParseOptions(invalid_row_handler=self.invalid_rows),
            convert_options=convert_options(),
        ).combine_chunks().to_batches()

    def batches(self):
        sock = self._connect()
        print(f"[source] connected to {self.address[0]}:{self.address[1]}", flush=True)
        buf = bytearray()
        while b"\n" not in buf:  # first line = CSV header
            data = sock.recv(65536)
            if not data:
                return
            buf += data
        nl = buf.index(b"\n")
        self.names = [c.strip() for c in buf[:nl].decode().rstrip("\r").split(",")] + ["_trailing"]
        del buf[: nl + 1]
        sock.setblocking(False)
        first_arrival = time.time() if buf else None
        last_flush = last_data = last_status = time.time()
        rows_at_status = 0
        eof = False
        while True:
            now = time.time()
            wait = max(0.0, min(last_flush + self.flush_s, last_data + self.idle_s) - now)
            readable, _, _ = select.select([sock], [], [], wait)
            now = time.time()
            if readable:
                try:
                    data = sock.recv(1 << 20)
                except BlockingIOError:
                    data = None
                except ConnectionResetError:  # feeder gone: treat as end of stream
                    data = b""
                if data:
                    if first_arrival is None:
                        first_arrival = now
                    buf += data
                    self.stats["bytes"] += len(data)
                    last_data = now
                elif data == b"":
                    eof = True
            due = now - last_flush >= self.flush_s or len(buf) >= self.block_size
            if buf and (due or eof):
                cut = len(buf) if eof else buf.rfind(b"\n") + 1
                if cut > 0:
                    chunk = bytes(buf[:cut])
                    del buf[:cut]
                    t0 = time.time()
                    batches = self._parse(chunk)
                    t1 = time.time()
                    self.stats["read_s"] += t1 - t0
                    max_before = self.max_t
                    for rb in batches:
                        parts, new = self.transform(rb)
                        self.stats["batches"] += 1
                        yield first_arrival, parts, new, self.watermark
                    if self.max_t != max_before:
                        self._clock_wall = now
                    self.stats["transform_s"] += time.time() - t1
                    self.stats["invalid_rows_skipped"] = self.invalid_rows.count
                    first_arrival = now if buf else None
                last_flush = now
            elif not buf:
                last_flush = now
            if eof:
                return
            if (now - last_data >= self.idle_s and self._clock_wall is not None
                    and self.max_t != _NO_TIME):
                clock = self.max_t + int((now - self._clock_wall) * 1000 * self.clock_rate)
                if clock > self.max_t:
                    self.max_t = clock
                    self._clock_wall = now
                    self.stats["idle_advances"] += 1
                    yield now, [None] * self.n_workers, [], self.watermark
                last_data = now  # next idle check after another idle_s
            if self.status_s and now - last_status >= self.status_s:
                rate = (self.stats["rows"] - rows_at_status) / (now - last_status)
                wm = self.watermark
                wm_s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(wm / 1000)) if wm else "-"
                print(f"[source] rows={self.stats['rows']:,} ({rate:,.0f}/s)  "
                      f"price={self.stats['price_events']:,}  watermark={wm_s}  "
                      f"late={self.stats['late_dropped']}  idle_adv={self.stats['idle_advances']}",
                      flush=True)
                last_status, rows_at_status = now, self.stats["rows"]
