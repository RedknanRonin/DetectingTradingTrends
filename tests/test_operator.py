"""Unit tests for the window/EMA/crossover operator (hand-crafted event sequences)."""
import numpy as np
import pytest

from tickstream.config import WINDOW_MS
from tickstream.operator import ADVICE_BUY, ADVICE_SELL, WindowEmaOperator

A38, A100 = 2 / 39, 2 / 101
W = WINDOW_MS


def ev(*rows):
    """rows of (sym, t_ms, price) -> arrays"""
    s, t, p = zip(*rows) if rows else ((), (), ())
    return np.array(s), np.array(t), np.array(p, dtype=float)


def ema_seq(closes, a):
    e, out = 0.0, []
    for c in closes:
        e = c * a + e * (1 - a)
        out.append(e)
    return out


def test_window_not_evaluated_until_watermark_passes_end():
    op = WindowEmaOperator(warmup_windows=0)
    r = op.process(*ev((0, 1000, 10.0), (0, 2000, 11.0)), watermark=W - 1, emit_all=True)
    assert len(r["sym"]) == 0
    r = op.process(*ev(), watermark=W, emit_all=True)  # w0 = [0, W) closes at wm = W
    assert r["sym"].tolist() == [0]
    assert r["close"][0] == 11.0  # last price in window


def test_ema_values_follow_spec_with_ema0_zero():
    op = WindowEmaOperator(warmup_windows=0)
    closes = [10.0, 12.0, 11.0, 15.0]
    rows = [(0, i * W + 10, c) for i, c in enumerate(closes)]
    r = op.process(*ev(*rows), watermark=np.inf, emit_all=True)
    np.testing.assert_allclose(r["ema38"], ema_seq(closes, A38))
    np.testing.assert_allclose(r["ema100"], ema_seq(closes, A100))
    assert r["win"].tolist() == [0, 1, 2, 3]


def test_close_is_latest_event_time_not_arrival_order():
    op = WindowEmaOperator(warmup_windows=0)
    # out of order within the window: event at t=5000 arrives after t=6000
    r = op.process(*ev((0, 6000, 20.0), (0, 5000, 19.0)), watermark=np.inf, emit_all=True)
    assert r["close"][0] == 20.0


def test_close_tie_on_time_takes_last_arrival():
    op = WindowEmaOperator(warmup_windows=0)
    r = op.process(*ev((0, 6000, 20.0), (0, 6000, 21.0)), watermark=np.inf, emit_all=True)
    assert r["close"][0] == 21.0


def test_empty_window_is_skipped_and_state_carries_over():
    op = WindowEmaOperator(warmup_windows=0)
    # prices in w0 and w3 only
    r = op.process(*ev((0, 10, 10.0), (0, 3 * W + 10, 20.0)), watermark=np.inf, emit_all=True)
    assert r["win"].tolist() == [0, 3]
    np.testing.assert_allclose(r["ema38"], ema_seq([10.0, 20.0], A38))


def test_events_split_over_batches_and_pending_state():
    op = WindowEmaOperator(warmup_windows=0)
    op.process(*ev((0, 10, 10.0)), watermark=100, emit_all=True)
    r = op.process(*ev((0, 20, 12.0), (0, W + 5, 13.0)), watermark=W + 5, emit_all=True)
    assert r["win"].tolist() == [0] and r["close"][0] == 12.0
    r = op.process(*ev(), watermark=np.inf, emit_all=True)
    assert r["win"].tolist() == [1] and r["close"][0] == 13.0


def test_no_signal_in_first_window_even_without_warmup():
    # literal formulas: EMA_0 = 0 -> EMA38 = 2/39*C > EMA100 = 2/101*C, prev 0 <= 0 -> "BUY"
    op = WindowEmaOperator(warmup_windows=0)
    r = op.process(*ev((0, 10, 100.0)), watermark=np.inf, emit_all=True)
    assert r["ema38"][0] > r["ema100"][0] and r["prev38"][0] == r["prev100"][0] == 0
    assert r["advice"][0] == 0  # fake crossover suppressed (D11)


def test_k0_equals_k1_real_crossovers_kept():
    # only the first window is affected: K=0 and K=1 give identical advisories, real ones kept
    rows = [(7, i * W + 1, c) for i, c in enumerate(_cross_series())]
    k0 = WindowEmaOperator(warmup_windows=0).process(*ev(*rows), watermark=np.inf, emit_all=True)
    k1 = WindowEmaOperator(warmup_windows=1).process(*ev(*rows), watermark=np.inf, emit_all=True)
    assert k0["advice"].tolist() == k1["advice"].tolist()
    assert set(k0["advice"][k0["advice"] != 0].tolist()) == {ADVICE_BUY, ADVICE_SELL}


def _cross_series():
    # long flat phase (EMAs converge), sharp fall (bearish cross), flat, rise (bullish cross)
    return ([100.0] * 300 + [100 - 3 * i for i in range(1, 26)] + [25.0] * 60
            + [25 + 4 * i for i in range(1, 40)])


def test_crossovers_match_bruteforce_definition():
    closes = _cross_series()
    op = WindowEmaOperator(warmup_windows=1)
    rows = [(7, i * W + 1, c) for i, c in enumerate(closes)]
    r = op.process(*ev(*rows), watermark=np.inf, emit_all=True)
    e38, e100 = ema_seq(closes, A38), ema_seq(closes, A100)
    expected = []
    for i in range(1, len(closes)):
        if e38[i] > e100[i] and e38[i - 1] <= e100[i - 1]:
            expected.append((i, ADVICE_BUY))
        elif e38[i] < e100[i] and e38[i - 1] >= e100[i - 1]:
            expected.append((i, ADVICE_SELL))
    got = [(int(w), int(a)) for w, a in zip(r["win"], r["advice"]) if a != 0]
    assert got == expected
    assert {a for _, a in got} == {ADVICE_BUY, ADVICE_SELL}


def test_warmup_counts_previous_windows():
    closes = _cross_series()
    rows = [(7, i * W + 1, c) for i, c in enumerate(closes)]
    full = WindowEmaOperator(warmup_windows=1).process(*ev(*rows), watermark=np.inf, emit_all=True)
    k = int(full["win"][full["advice"] != 0][0]) + 1  # suppress the first crossover
    late = WindowEmaOperator(warmup_windows=k).process(*ev(*rows), watermark=np.inf, emit_all=True)
    assert (late["advice"] != 0).sum() == (full["advice"] != 0).sum() - 1


def test_symbols_are_independent_and_subscription_filters_output():
    op = WindowEmaOperator(warmup_windows=0)
    op.subscribe([1])
    r = op.process(*ev((0, 10, 5.0), (1, 20, 7.0), (0, W + 1, 6.0)), watermark=np.inf)
    assert r["sym"].tolist() == [1]
    np.testing.assert_allclose(r["ema38"], [7.0 * A38])


def test_late_event_for_evaluated_window_is_dropped():
    op = WindowEmaOperator(warmup_windows=0)
    op.process(*ev((0, 10, 5.0)), watermark=W, emit_all=True)
    r = op.process(*ev((0, 20, 99.0)), watermark=W, emit_all=True)
    assert len(r["sym"]) == 0 and op.late_dropped == 1


def test_multiple_windows_closed_at_once_are_evaluated_in_order():
    op = WindowEmaOperator(warmup_windows=0)
    rows = [(0, 3 * W + 1, 13.0), (0, 1, 10.0), (0, W + 1, 11.0), (1, 2 * W + 1, 50.0)]
    r = op.process(*ev(*rows), watermark=np.inf, emit_all=True)
    sym0 = r["sym"] == 0
    assert r["win"][sym0].tolist() == [0, 1, 3]
    np.testing.assert_allclose(r["ema38"][sym0], ema_seq([10.0, 11.0, 13.0], A38))


def test_state_grows_for_large_symbol_ids():
    op = WindowEmaOperator(warmup_windows=0, capacity=4)
    r = op.process(*ev((10_000, 1, 3.0)), watermark=np.inf, emit_all=True)
    assert r["sym"].tolist() == [10_000]


# ---------------------------------------------------------------- 1-minute OHLC bars
from tickstream.operator import MinuteBars  # noqa: E402

M = 60_000


def test_minute_bars_ohlc_and_emission_by_watermark():
    b = MinuteBars()
    r = b.process(*ev((0, 1_000, 10.0), (0, 20_000, 12.0), (0, 30_000, 9.0), (0, 59_000, 11.0),
                      (0, M + 5, 20.0)), watermark=M - 1)
    assert len(r["sym"]) == 0  # minute 0 not yet complete
    r = b.process(*ev(), watermark=M)
    assert r["minute"].tolist() == [0]
    assert (r["open"][0], r["high"][0], r["low"][0], r["close"][0], r["n"][0]) == (10.0, 12.0, 9.0, 11.0, 4)
    r = b.process(*ev(), watermark=np.inf)
    assert r["minute"].tolist() == [1] and r["open"][0] == r["close"][0] == 20.0


def test_minute_bars_out_of_order_and_late():
    b = MinuteBars()
    r = b.process(*ev((1, 50_000, 5.0), (1, 10_000, 4.0)), watermark=M)  # arrival out of order
    assert (r["open"][0], r["close"][0]) == (4.0, 5.0)
    b.process(*ev((1, 30_000, 99.0)), watermark=M)  # minute 0 already emitted -> late
    assert b.late_dropped == 1


def test_window_close_equals_last_minute_bar_close():
    rng = np.random.default_rng(1)
    t = np.sort(rng.integers(0, 3 * W, 500))
    sym = rng.integers(0, 3, 500)
    price = rng.uniform(10, 20, 500)
    op = WindowEmaOperator(warmup_windows=0)
    r = op.process(sym, t, price, watermark=np.inf, emit_all=True)
    bars = MinuteBars().process(sym, t, price, watermark=np.inf)
    for s, w, c in zip(r["sym"], r["win"], r["close"]):
        sel = (bars["sym"] == s) & (bars["minute"] * M // W == w)
        assert bars["close"][sel][-1] == c
