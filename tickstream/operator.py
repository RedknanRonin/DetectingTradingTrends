"""Keyed window operator: per-symbol Close per tumbling window, EMA38/EMA100, crossovers.

Pure computation, no I/O, so it can be unit-tested and reused by every worker.
State is held in numpy arrays indexed by a global integer symbol id.

Per batch the operator
  1. ingests price events (sym, t, price) of its partition,
  2. reduces them to the last price per (symbol, window) ("Close"), kept as *pending*,
  3. evaluates every pending window that the watermark has passed (window end <= wm),
     in window order: EMA update (Query 1) and crossover detection (Query 2).
Windows without price events for a symbol are skipped: its EMA state carries over (D6).
"""
import numpy as np

from .config import EMA_J, WINDOW_MS

ADVICE_NONE, ADVICE_BUY, ADVICE_SELL = 0, 1, -1


class WindowEmaOperator:
    def __init__(self, warmup_windows: int = 38, j=EMA_J, window_ms: int = WINDOW_MS,
                 capacity: int = 8192):
        self.window_ms = window_ms
        self.warmup = warmup_windows
        self.a1 = 2.0 / (1 + j[0])
        self.a2 = 2.0 / (1 + j[1])
        self.ema1 = np.zeros(capacity)                 # EMA_{s, last evaluated window}; EMA_0 = 0
        self.ema2 = np.zeros(capacity)
        self.nwin = np.zeros(capacity, dtype=np.int64)  # windows evaluated so far per symbol
        self.subscribed = np.zeros(capacity, dtype=bool)
        self.events = np.zeros(capacity, dtype=np.int64)  # price events seen per symbol
        # pending (not yet evaluated) closes: one row per (symbol, window)
        self.p_sym = np.empty(0, np.int64)
        self.p_win = np.empty(0, np.int64)
        self.p_t = np.empty(0, np.int64)
        self.p_price = np.empty(0, np.float64)
        self.closed_upto = None  # windows with index < closed_upto are evaluated
        self.late_dropped = 0
        self.last_evaluated = 0

    # ---------------------------------------------------------------- state management
    def _ensure(self, max_sym: int) -> None:
        if max_sym < len(self.ema1):
            return
        n = max(max_sym + 1, 2 * len(self.ema1))
        for name in ("ema1", "ema2", "nwin", "subscribed", "events"):
            old = getattr(self, name)
            new = np.zeros(n, dtype=old.dtype)
            new[: len(old)] = old
            setattr(self, name, new)

    def subscribe(self, sym_ids) -> None:
        sym_ids = np.asarray(sym_ids, dtype=np.int64)
        if len(sym_ids):
            self._ensure(int(sym_ids.max()))
            self.subscribed[sym_ids] = True

    # ---------------------------------------------------------------- processing
    def process(self, sym, t, price, watermark, emit_all: bool = False):
        """Ingest events and evaluate windows closed by `watermark` (ms, or None for none).

        Returns a dict of numpy arrays with one row per evaluated (symbol, window) that is
        emitted (subscribed symbols, or all if emit_all):
            sym, win, close, ema38, ema100, prev38, prev100, advice
        """
        sym = np.asarray(sym, dtype=np.int64)
        if len(sym):
            self._ensure(int(sym.max()))
        t = np.asarray(t, dtype=np.int64)
        price = np.asarray(price, dtype=np.float64)
        if len(sym):
            self.events += np.bincount(sym, minlength=len(self.events))
        win = t // self.window_ms
        if self.closed_upto is not None:  # defensive: never re-open an evaluated window
            ok = win >= self.closed_upto
            if not ok.all():
                self.late_dropped += int((~ok).sum())
                sym, t, price, win = sym[ok], t[ok], price[ok], win[ok]

        # 1+2) merge with pending and reduce to the last price per (symbol, window).
        a_sym = np.concatenate((self.p_sym, sym))
        a_win = np.concatenate((self.p_win, win))
        a_t = np.concatenate((self.p_t, t))
        a_price = np.concatenate((self.p_price, price))
        if len(a_sym):
            # lexsort is stable: ties in t keep arrival order, so the last row of each
            # (sym, win) group is the latest price event observed in that window.
            order = np.lexsort((a_t, a_win, a_sym))
            a_sym, a_win, a_t, a_price = a_sym[order], a_win[order], a_t[order], a_price[order]
            last = np.ones(len(a_sym), dtype=bool)
            last[:-1] = (a_sym[1:] != a_sym[:-1]) | (a_win[1:] != a_win[:-1])
            a_sym, a_win, a_t, a_price = a_sym[last], a_win[last], a_t[last], a_price[last]

        # 3) evaluate windows the watermark has passed.
        if watermark is None:  # no watermark progress: evaluate nothing new
            close_before = self.closed_upto if self.closed_upto is not None else np.iinfo(np.int64).min
        elif watermark == np.inf:  # end of stream: flush everything
            close_before = np.iinfo(np.int64).max
        else:
            close_before = int(watermark) // self.window_ms
            if self.closed_upto is not None:
                close_before = max(close_before, self.closed_upto)
        self.closed_upto = close_before

        closing = a_win < close_before
        self.p_sym, self.p_win, self.p_t, self.p_price = (
            a_sym[~closing], a_win[~closing], a_t[~closing], a_price[~closing])
        c_sym, c_win, c_price = a_sym[closing], a_win[closing], a_price[closing]
        self.last_evaluated = len(c_sym)  # (symbol, window) pairs evaluated in this call
        if not len(c_sym):
            return _empty_result()

        # A symbol may have several closing windows; evaluate them in window order.
        order = np.lexsort((c_sym, c_win))
        c_sym, c_win, c_price = c_sym[order], c_win[order], c_price[order]
        out = []
        bounds = np.flatnonzero(np.diff(c_win)) + 1
        for s, w, p in zip(np.split(c_sym, bounds), np.split(c_win, bounds), np.split(c_price, bounds)):
            out.append(self._evaluate(s, w, p, emit_all))
        return {k: np.concatenate([o[k] for o in out]) for k in out[0]}

    def _evaluate(self, sym, win, close, emit_all):
        """EMA update + crossover detection for one window; `sym` are unique."""
        prev1, prev2 = self.ema1[sym], self.ema2[sym]
        new1 = close * self.a1 + prev1 * (1.0 - self.a1)
        new2 = close * self.a2 + prev2 * (1.0 - self.a2)
        self.ema1[sym], self.ema2[sym] = new1, new2
        eligible = self.nwin[sym] >= self.warmup  # D11 warm-up (K previous windows)
        self.nwin[sym] += 1
        buy = (new1 > new2) & (prev1 <= prev2) & eligible     # Eq. (1),(2)
        sell = (new1 < new2) & (prev1 >= prev2) & eligible    # Eq. (3),(4)
        advice = np.where(buy, ADVICE_BUY, np.where(sell, ADVICE_SELL, ADVICE_NONE)).astype(np.int8)
        keep = slice(None) if emit_all else self.subscribed[sym]
        return dict(sym=sym[keep], win=win[keep], close=close[keep], ema38=new1[keep],
                    ema100=new2[keep], prev38=prev1[keep], prev100=prev2[keep],
                    advice=advice[keep])


class MinuteBars:
    """1-minute OHLC bars for subscribed symbols (visualisation only, not part of Q1/Q2).

    Raw price events of subscribed symbols are buffered until the watermark passes the end
    of their minute; then each (symbol, minute) is aggregated to open/high/low/close/count.
    Buffer size is bounded by (subscribed symbols x events per minute + lateness).
    """

    def __init__(self, bar_ms: int = 60_000):
        self.bar_ms = bar_ms
        self.b_sym = np.empty(0, np.int64)
        self.b_t = np.empty(0, np.int64)
        self.b_price = np.empty(0, np.float64)
        self.closed_upto = None  # minutes with index < closed_upto are emitted
        self.late_dropped = 0

    def process(self, sym, t, price, watermark):
        sym = np.asarray(sym, dtype=np.int64)
        t = np.asarray(t, dtype=np.int64)
        price = np.asarray(price, dtype=np.float64)
        if self.closed_upto is not None and len(t):
            ok = t // self.bar_ms >= self.closed_upto  # never re-open an emitted bar
            self.late_dropped += int((~ok).sum())
            sym, t, price = sym[ok], t[ok], price[ok]
        a_sym = np.concatenate((self.b_sym, sym))
        a_t = np.concatenate((self.b_t, t))
        a_price = np.concatenate((self.b_price, price))

        if watermark is None:
            return _empty_bars()
        close_before = (np.iinfo(np.int64).max if watermark == np.inf
                        else int(watermark) // self.bar_ms)
        if self.closed_upto is not None:
            close_before = max(close_before, self.closed_upto)
        self.closed_upto = close_before

        minute = a_t // self.bar_ms
        done = minute < close_before
        self.b_sym, self.b_t, self.b_price = a_sym[~done], a_t[~done], a_price[~done]
        if not done.any():
            return _empty_bars()
        s, m, tt, p = a_sym[done], minute[done], a_t[done], a_price[done]
        order = np.lexsort((tt, m, s))  # stable: ties in time keep arrival order
        s, m, p = s[order], m[order], p[order]
        start = np.flatnonzero(np.r_[True, (s[1:] != s[:-1]) | (m[1:] != m[:-1])])
        end = np.r_[start[1:], len(s)] - 1
        return dict(sym=s[start], minute=m[start], open=p[start], close=p[end],
                    high=np.maximum.reduceat(p, start), low=np.minimum.reduceat(p, start),
                    n=(end - start + 1).astype(np.int64))


def _empty_bars():
    return dict(sym=np.empty(0, np.int64), minute=np.empty(0, np.int64), open=np.empty(0),
                close=np.empty(0), high=np.empty(0), low=np.empty(0), n=np.empty(0, np.int64))


def _empty_result():
    return dict(sym=np.empty(0, np.int64), win=np.empty(0, np.int64), close=np.empty(0),
                ema38=np.empty(0), ema100=np.empty(0), prev38=np.empty(0),
                prev100=np.empty(0), advice=np.empty(0, np.int8))
