"""Unit tests for parsing, normalization, late filtering and partitioning in the source."""
import numpy as np
import pyarrow as pa

from tickstream.config import WINDOW_MS
from tickstream.source import Partitioner, Source, balanced_assignment, time_of_day_ms

HEADER_COMMENTS = "".join(f"# comment {i}\n" for i in range(11))
HEADER = ("ID,SecType,Date,Time,Ask,Ask volume,Bid,Bid volume,Ask time,Day's high ask,Close,Currency,"
          "Day's high ask time,Day's high,ISIN,Auction price,Day's low ask,Day's low,Day's low ask time,"
          "Open,Nominal value,Last,Last volume,Trading time,Total volume,Mid price,Trading date,Profit,"
          "Current price,Related indices,Day high bid time, Day low bid time,Open Time, Last trade time,"
          " Close Time, Day high Time, Day low Time,Bid time,Auction Time\n")
DESC = "#'Unique ID','Security Type'\n"


def row(sym, last="", ttime="", tdate="", date="08-11-2021", sectype="E", systime=None):
    f = [""] * 40  # 39 columns + trailing comma, like the real files
    # arrival (system) time defaults to the trading time truncated to seconds, as in the data
    systime = systime or ((ttime[:8] + ".000") if ttime else "09:00:00.000")
    f[0], f[1], f[2], f[3] = sym, sectype, date, systime
    f[21], f[23], f[26] = last, ttime, tdate
    return ",".join(f) + "\n"


def write_csv(tmp_path, rows, name="day.csv"):
    p = tmp_path / name
    p.write_text(HEADER_COMMENTS + HEADER + DESC + "".join(rows))
    return str(p)


def collect(src):
    out = []
    for _, parts, _, wm in src.batches():
        for w, part in enumerate(parts):
            if part is not None:
                for s, t, pr in zip(*part):
                    out.append((w, src.registry.names[s], int(t), float(pr)))
    return out


def test_time_of_day_parsing():
    tt = pa.array(["00:00:00.000", "09:30:15.250", "23:59:59.999", "10:00:01.5"])
    assert time_of_day_ms(tt).tolist() == [0, 34215250, 86399999, 36001500]


def test_price_filter_and_event_time(tmp_path):
    day = 1636329600000  # 08-11-2021 00:00 as naive epoch ms
    path = write_csv(tmp_path, [
        row("A.FR"),                                         # housekeeping, no payload
        row("A.FR", last="0.000000", ttime="00:00:00.000"),  # placeholder price
        row("A.FR", last="10.5", ttime="09:00:00.100"),      # date from system Date
        row("B.NL", last="7.25", ttime="09:00:01.000", tdate="08-11-2021"),
    ])
    src = Source([path], 1, Partitioner(1), lateness_ms=1000, block_size=1 << 20)
    got = collect(src)
    assert got == [(0, "A.FR", day + 9 * 3600_000 + 100, 10.5),
                   (0, "B.NL", day + 9 * 3600_000 + 1000, 7.25)]
    assert src.stats["rows"] == 4 and src.stats["price_events"] == 2
    assert src.stats["non_price_events"] == 2


def test_late_events_dropped_against_running_watermark(tmp_path):
    # window boundary at 09:05:00. lateness 500 ms.
    rows = [
        row("A.FR", last="1", ttime="09:04:59.000"),
        row("B.FR", last="2", ttime="09:05:00.800"),  # watermark -> 09:05:00.300: w(09:00) closed
        row("C.FR", last="3", ttime="09:04:59.900"),  # late -> dropped
        row("D.FR", last="4", ttime="09:05:00.100"),  # out of order, but its window still open
    ]
    src = Source([write_csv(tmp_path, rows)], 1, Partitioner(1), lateness_ms=500, block_size=1 << 20)
    assert [g[1] for g in collect(src)] == ["A.FR", "B.FR", "D.FR"]
    assert src.stats["late_dropped"] == 1
    src2 = Source([write_csv(tmp_path, rows, "b.csv")], 1, Partitioner(1), lateness_ms=1000,
                  block_size=1 << 20)
    assert len(collect(src2)) == 4 and src2.stats["late_dropped"] == 0


def test_hash_partitioning_is_deterministic_and_keeps_symbol_order(tmp_path):
    rows = [row(f"S{i % 7}.FR", last=str(i + 1), ttime=f"09:00:{i:02d}.000") for i in range(40)]
    path = write_csv(tmp_path, rows)
    a = collect(Source([path], 3, Partitioner(3), 1000, 1 << 20))
    b = collect(Source([path], 3, Partitioner(3), 1000, 1 << 20))
    assert a == b
    owner = {}
    for w, s, t, _ in a:
        assert owner.setdefault(s, w) == w  # a symbol always goes to the same worker
    for s in owner:
        ts = [t for _, s2, t, _ in a if s2 == s]
        assert ts == sorted(ts)


def test_balanced_assignment_is_lpt():
    w = {"a": 10, "b": 6, "c": 5, "d": 4, "e": 1}
    m = balanced_assignment(w, 2)
    loads = [sum(v for k, v in w.items() if m[k] == p) for p in range(2)]
    # LPT: a->0, b->1, c->1, d->0, e->1  (greedy, not necessarily optimal [13, 13])
    assert loads == [14, 12]


def test_window_alignment_midnight():
    day = 1636329600000
    assert day % WINDOW_MS == 0  # w0 starts at 00:00


def test_stale_trading_timestamp_dropped_not_late(tmp_path):
    rows = [
        row("A.FR", last="1", ttime="18:55:59.000"),
        row("PX1.FR", last="7046.17", ttime="01:00:00.000", systime="18:56:11.000", sectype="I"),
        row("B.FR", last="2", ttime="18:56:12.500"),
    ]
    src = Source([write_csv(tmp_path, rows)], 1, Partitioner(1), 1000, 1 << 20)
    assert [g[1] for g in collect(src)] == ["A.FR", "B.FR"]
    assert src.stats["stale_timestamp_dropped"] == 1 and src.stats["late_dropped"] == 0
    off = Source([write_csv(tmp_path, rows, "b.csv")], 1, Partitioner(1), 1000, 1 << 20, stale_ms=None)
    collect(off)
    assert off.stats["late_dropped"] == 1  # without the rule it is classified as late instead
