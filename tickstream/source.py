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
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .config import WINDOW_MS
from .schema import COL_DATE, COL_ID, COL_LAST, COL_SECTYPE, COL_TRADING_DATE, COL_TRADING_TIME

HEADER_SKIP_ROWS = 11  # comment lines before the header; one '#...' description line after it
SOURCE_COLUMNS = [COL_ID, COL_SECTYPE, COL_DATE, COL_LAST, COL_TRADING_TIME, COL_TRADING_DATE]
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
        convert_options=pacsv.ConvertOptions(
            include_columns=SOURCE_COLUMNS,
            strings_can_be_null=True,  # empty field -> null (needed for coalesce / validity)
            column_types={c: pa.string() for c in SOURCE_COLUMNS if c != COL_LAST}
            | {COL_LAST: pa.float64()},
        ),
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
                 block_size: int, max_rows=None, use_threads=True):
        self.files = files
        self.n_workers = n_workers
        self.part = partitioner
        self.lateness = lateness_ms
        self.block_size = block_size
        self.max_rows = max_rows
        self.use_threads = use_threads
        self.registry = SymbolRegistry()
        self.max_t = _NO_TIME  # max event time seen so far
        self.day_cache: dict[str, int] = {}
        self.invalid_rows = InvalidRowCounter()
        self.stats = dict(rows=0, price_events=0, non_price_events=0, late_dropped=0,
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

        # event time
        date = pc.coalesce(rb.column(COL_TRADING_DATE), rb.column(COL_DATE))
        denc = pc.dictionary_encode(date)
        for d in denc.dictionary.to_pylist():
            if d not in self.day_cache:
                self.day_cache[d] = _day_ms(d)
        day = np.array([self.day_cache[d] for d in denc.dictionary.to_pylist()], dtype=np.int64)
        t = day[denc.indices.to_numpy(zero_copy_only=False)] + time_of_day_ms(rb.column(COL_TRADING_TIME))

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
