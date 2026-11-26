"""Trading-trend dashboard (D8).  Run:  streamlit run dashboard/app.py

Views
  * Signal board  - one tile per subscribed symbol: current trend (EMA38 vs EMA100), the
                    EMA spread in %, and the latest advisory, sorted so that fresh
                    crossovers stand out ("cut through the noise").
  * Symbol detail - 1-minute OHLC candles (subscribed symbols) with EMA38/EMA100 per
                    window plotted at the window end, BUY/SELL markers, shaded
                    bullish/bearish regimes, and the EMA spread panel.
  * Advisory feed - newest advisories first; live-tails advisories_live.jsonl while a run
                    is in progress.
  * Run metrics   - throughput, latency, CPU and per-worker load of the selected run.
  * Replay        - sidebar controls (start / previous / play-pause / next / end, a time
                    slider, step = 1 minute or 5 minutes, speed): every view shows the state
                    "as of" the selected time, so the week can be replayed; advisories from
                    the last 5 minutes are highlighted.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
FIVE_MIN, ONE_MIN = timedelta(minutes=5), timedelta(minutes=1)
GREEN, RED, BLUE, ORANGE, GREY = "#1a9850", "#d73027", "#2c7bb6", "#fdae61", "#888888"

st.set_page_config(page_title="Trading Trends", layout="wide")


def find_runs():
    """Runs with output, newest first; subscription runs before --emit-all validation runs."""
    def emit_all(run):
        try:
            return json.loads((run / "metrics.json").read_text())["config"]["emit_all"]
        except (OSError, KeyError, ValueError):
            return False
    runs = {p.parent for p in RUNS.rglob("windows.parquet")}
    return sorted(runs, key=lambda p: (emit_all(p), -p.stat().st_mtime))


@st.cache_data(show_spinner=False)
def load_windows(path: str, mtime: float) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(
        spread_pct=(pl.col("ema38") - pl.col("ema100")) / pl.col("ema100") * 100)


@st.cache_data(show_spinner=False)
def _read_bars(path: str, mtime: float) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_bars(run: Path):
    p = run / "bars.parquet"
    return _read_bars(str(p), p.stat().st_mtime) if p.exists() else None


def load_live(run: Path) -> pl.DataFrame:
    p = run / "advisories_live.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return pl.DataFrame()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return pl.DataFrame(rows)


runs = find_runs()
if not runs:
    st.warning(f"No runs found under {RUNS}. Run `python -m tickstream ...` first.")
    st.stop()

with st.sidebar:
    st.header("Run")
    run = st.selectbox("Output directory", runs, format_func=lambda p: str(p.relative_to(RUNS)))
    wpath = run / "windows.parquet"
    df = load_windows(str(wpath), wpath.stat().st_mtime)
    if df.is_empty():
        st.info("This run emitted no window results (no subscribed symbols?).")
        st.stop()
    sectypes = st.multiselect("Security type", ["E", "I"], default=["E", "I"],
                              format_func=lambda s: {"E": "Equity", "I": "Index"}[s])
    df = df.filter(pl.col("sectype").is_in(sectypes))
    t_min, t_max = df["window_start"].min(), df["window_start"].max()
    st.caption(f"{df['symbol'].n_unique()} symbols · windows {t_min:%a %d.%m %H:%M} – {t_max:%a %d.%m %H:%M}")

    # ---------------------------------------------------------------- replay controls
    # A time cursor ("now"): every view shows what was known at that moment, i.e. windows
    # whose end <= now and 1-minute bars whose end <= now. Times without data (nights,
    # weekends) are skipped. Play advances the cursor on a timer.
    st.header("Replay")
    bars_all = load_bars(run)
    steps_avail = ["5 minutes (windows)"] + (["1 minute (candles)"] if bars_all is not None else [])
    step = st.radio("Step", steps_avail, horizontal=True, index=len(steps_avail) - 1)
    win_ends = (df["window_start"] + FIVE_MIN).unique().sort()
    if step.startswith("1"):
        bar_ends = (bars_all.filter(pl.col("symbol").is_in(df["symbol"].unique().to_list()))["minute_start"]
                    + ONE_MIN)
        times = pl.concat([win_ends, bar_ends]).unique().sort().to_list()
    else:
        times = win_ends.to_list()
    ss = st.session_state
    # ss.pos (a datetime) is the source of truth; the slider widget mirrors it (a widget's
    # own state may not be changed by code after the widget is created, e.g. by the timer).
    if ss.get("replay_run") != str(run) or ss.get("pos") is None:
        ss.replay_run, ss.pos, ss.playing = str(run), times[-1], False
    if ss.pos not in times:  # step size changed: snap to the latest time <= pos
        earlier = [t for t in times if t <= ss.pos]
        ss.pos = earlier[-1] if earlier else times[0]
    idx = times.index(ss.pos)

    def _move(to_idx):
        ss.pos = times[max(0, min(len(times) - 1, to_idx))]

    def _toggle_play():
        if not ss.playing and ss.pos == times[-1]:
            ss.pos = times[0]  # pressing play at the end restarts the replay
        ss.playing = not ss.playing
        ss.last_tick = 0.0

    unit = "minute" if step.startswith("1") else "window"
    b = st.columns(5)
    b[0].button("⏮", on_click=_move, args=(0,), help="start", width="stretch")
    b[1].button("◀", on_click=_move, args=(idx - 1,), help=f"previous {unit}", width="stretch")
    b[2].button("⏸" if ss.playing else "▶", on_click=_toggle_play, help="play / pause",
                type="primary", width="stretch")
    b[3].button("▶▶", on_click=_move, args=(idx + 1,), help=f"next {unit}", width="stretch")
    b[4].button("⏭", on_click=_move, args=(len(times) - 1,), help="end (show all)",
                width="stretch")
    ss.pos_slider = ss.pos

    def _slid():
        ss.pos = ss.pos_slider

    st.select_slider("Time", options=times, key="pos_slider", on_change=_slid,
                     format_func=lambda t: f"{t:%a %d.%m %H:%M}")
    speed = st.select_slider(f"Speed ({unit}s per second)", options=[1, 2, 5, 12, 36], value=2)

    import time as _time

    @st.fragment(run_every=0.25 if ss.playing else None)
    def _ticker():
        # Advance only on timer ticks: the fragment also runs as part of every full rerun,
        # so the elapsed-time check prevents a rerun from immediately advancing again.
        if not ss.playing:
            return
        now_s = _time.time()
        last = ss.get("last_tick", 0.0) or now_s - 1.0 / speed
        n = int((now_s - last) * speed)  # steps due since the last advance
        if n < 1:
            return
        ss.last_tick = now_s
        i = min(times.index(ss.pos) + n, len(times) - 1)
        ss.pos = times[i]
        if i == len(times) - 1:
            ss.playing = False
        st.rerun(scope="app")

    _ticker()
    st.caption(f"{unit.capitalize()} {idx + 1} of {len(times)}" + (" · playing" if ss.playing else ""))

df_all = df
now = ss.pos
df = df.filter(pl.col("window_start") + FIVE_MIN <= now)  # windows evaluated by "now"
bars = (bars_all.filter(pl.col("minute_start") + ONE_MIN <= now) if bars_all is not None else None)
fresh_after = now - FIVE_MIN  # advisories from windows that ended in the last 5 minutes are "new"

tab_board, tab_symbol, tab_feed, tab_metrics = st.tabs(
    ["Signal board", "Symbol detail", "Advisory feed", "Run metrics"])

# ------------------------------------------------------------------ signal board
with tab_board:
    latest = df.sort("win").group_by("symbol").agg(
        pl.col("sectype").last(), pl.col("window_start").last().alias("last_window"),
        pl.col("close").last(), pl.col("ema38").last(), pl.col("ema100").last(),
        pl.col("spread_pct").last(), pl.len().alias("windows"),
        (pl.col("advice") == 1).sum().alias("buys"), (pl.col("advice") == -1).sum().alias("sells"),
    )
    last_adv = (df.filter(pl.col("advice") != 0).sort("win").group_by("symbol")
                .agg(pl.col("advice").last().alias("last_advice"),
                     pl.col("window_start").last().alias("last_advice_at")))
    board = latest.join(last_adv, on="symbol", how="left").sort(
        ["last_advice_at", "symbol"], descending=[True, False], nulls_last=True)

    fresh = df.filter((pl.col("window_start") + FIVE_MIN > fresh_after) & (pl.col("advice") != 0))
    st.markdown(f"**State as of {now:%a %d.%m %H:%M}** · "
                f"{fresh.height} new advisor{'y' if fresh.height == 1 else 'ies'} in the last 5 minutes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols", board.height)
    c2.metric("Bullish now (EMA38 > EMA100)", int((board["spread_pct"] > 0).sum()))
    c3.metric("BUY advisories", int(board["buys"].sum()))
    c4.metric("SELL advisories", int(board["sells"].sum()))

    o1, o2 = st.columns([2, 1])
    sort_by = o1.radio("Sort tiles by", ["Most recent advisory", "Largest EMA spread", "Most windows"],
                       horizontal=True)
    n_tiles = o2.slider("Tiles shown", 10, 200, 40, step=10)
    key = {"Most recent advisory": (["last_advice_at", "symbol"], [True, False]),
           "Largest EMA spread": (["abs_spread"], [True]),
           "Most windows": (["windows"], [True])}[sort_by]
    shown = (board.with_columns(abs_spread=pl.col("spread_pct").abs())
             .sort(key[0], descending=key[1], nulls_last=True).head(n_tiles))

    # One HTML grid for all tiles: a single Streamlit element instead of one per symbol
    # (thousands of elements made the page stall before the other tabs were rendered).
    tiles = []
    for r in shown.iter_rows(named=True):
        bull = r["spread_pct"] > 0
        adv = {1: "&#9650; BUY", -1: "&#9660; SELL"}.get(r["last_advice"], "&ndash;")
        adv_col = {1: GREEN, -1: RED}.get(r["last_advice"], GREY)
        when = f"{r['last_advice_at']:%a %H:%M}" if r["last_advice_at"] else ""
        new = (r["last_advice_at"] is not None
               and r["last_advice_at"] + FIVE_MIN > fresh_after)  # raised in the last 5 min
        ring = f"box-shadow:0 0 0 2px {adv_col};" if new else ""
        badge = (f' <small style="background:{adv_col};color:white;padding:0 4px;border-radius:3px">'
                 f'NEW</small>') if new else ""
        tiles.append(
            f'<div style="border-left:6px solid {GREEN if bull else RED};padding:6px 10px;{ring}'
            f'border-radius:4px;background:rgba(127,127,127,0.08)">'
            f'<b>{r["symbol"]}</b> <small>{"Index" if r["sectype"] == "I" else "Equity"}</small>{badge}<br>'
            f'<span style="font-size:1.25em">{r["close"]:.4g}</span>'
            f'<span style="color:{GREEN if bull else RED}">&nbsp;{r["spread_pct"]:+.2f}%</span><br>'
            f'<small style="color:{adv_col}">{adv} {when}</small></div>')
    st.markdown('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));'
                'gap:10px">' + "".join(tiles) + "</div>", unsafe_allow_html=True)
    st.caption(f"Showing {shown.height} of {board.height} symbols. Border colour = current regime "
               "(green: EMA38 above EMA100). Percentage = EMA spread (EMA38 &minus; EMA100)/EMA100.")

# ------------------------------------------------------------------ symbol detail
with tab_symbol:
    # options from the whole run, so the selection stays stable while the replay advances
    # deterministic order (ties broken by name): if the option list changed between reruns,
    # Streamlit would treat it as a new widget and reset the selection
    syms = (df_all.group_by("symbol").len()
            .sort(["len", "symbol"], descending=[True, False])["symbol"].to_list())
    c_sym, c_fit = st.columns([3, 1])
    sym = c_sym.selectbox("Symbol (sorted by number of windows)", syms, key="symbol_select")
    fit_price = c_fit.toggle("Fit y-axis to price", value=True,
                             help="Off: show the full EMA range, including the start-up from EMA_0 = 0")
    d_all = df_all.filter(pl.col("symbol") == sym)
    d = df.filter(pl.col("symbol") == sym).sort("win")
    sb = (bars.filter(pl.col("symbol") == sym).sort("minute") if bars is not None else None)
    has_candles = sb is not None and not sb.is_empty()
    show_chart = not (d.is_empty() and not has_candles)
    if not show_chart:
        st.info(f"No data for {sym} yet at {now:%a %d.%m %H:%M} "
                f"(its first window ends {d_all['window_start'].min() + FIVE_MIN:%a %H:%M}).")
    # Window-level values are plotted at the window END: that is when Close/EMAs/advisories
    # become known (a window is evaluated once the next one starts), and it lines them up
    # with the 1-minute candles of the same period.
    d = d.with_columns(window_end=pl.col("window_start") + FIVE_MIN)
    x = d["window_end"].to_list()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)
    # shade bullish / bearish regimes (a regime holds from one window end to the next)
    bull = (d["ema38"] > d["ema100"]).to_numpy()
    if len(bull):
        starts = np.flatnonzero(np.r_[True, bull[1:] != bull[:-1]])
        ends = np.r_[starts[1:], len(bull)] - 1
        for s, e in zip(starts, ends):
            fig.add_vrect(x0=x[s], x1=x[e] + FIVE_MIN if e == len(bull) - 1 else x[e + 1],
                          fillcolor=GREEN if bull[s] else RED, opacity=0.06, line_width=0, row=1, col=1)
    if has_candles:
        fig.add_trace(go.Candlestick(
            x=sb["minute_start"], open=sb["open"], high=sb["high"], low=sb["low"], close=sb["close"],
            name="1-min OHLC", increasing_line_color=GREY, decreasing_line_color="#444444",
            increasing_fillcolor="rgba(136,136,136,0.35)", decreasing_fillcolor="rgba(68,68,68,0.7)",
            line_width=1), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=d["close"], name="Close (last price per window)",
                                 mode="lines", line=dict(color=GREY, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema38"], name="EMA38", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema100"], name="EMA100", line=dict(color=ORANGE, width=2)), row=1, col=1)
    for code, name, color, symbol in ((1, "BUY", GREEN, "triangle-up"), (-1, "SELL", RED, "triangle-down")):
        a = d.filter(pl.col("advice") == code)
        fig.add_trace(go.Scatter(x=a["window_end"], y=a["close"], mode="markers", name=name,
                                 marker=dict(color=color, size=13, symbol=symbol,
                                             line=dict(color="white", width=1))), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=d["spread_pct"], name="EMA spread %",
                         marker_color=np.where(bull, GREEN, RED)), row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04), xaxis_rangeslider_visible=False)
    # hide nights and weekends (no trading) so the trading sessions sit next to each other
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[22, 7], pattern="hour")])
    # fixed time axis = the symbol's whole run, so the series grow during replay
    fig.update_xaxes(range=[d_all["window_start"].min(), d_all["window_start"].max() + FIVE_MIN])
    if now < d_all["window_start"].max() + FIVE_MIN:
        fig.add_vline(x=now, line_width=1, line_dash="dot", line_color=GREY)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="spread %", row=2, col=1)
    if fit_price:
        # Price-focused y-axis: EMAs start at 0 (EMA_0 = 0) and would otherwise squash the
        # candles into a flat line until they converge; they enter the view once they do.
        lo = sb["low"].min() if has_candles else d["close"].min()
        hi = sb["high"].max() if has_candles else d["close"].max()
        if lo is not None and hi is not None:
            pad = max((hi - lo) * 0.08, abs(hi) * 0.002)
            fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    if show_chart:
        st.plotly_chart(fig, width="stretch")
    st.dataframe(d.filter(pl.col("advice") != 0).select(
        ["window_start", "window_end", "advice", "close", "ema38", "ema100", "prev38", "prev100"]),
        width="stretch")

# ------------------------------------------------------------------ advisory feed
with tab_feed:
    @st.fragment(run_every=2)
    def live_feed():
        live = load_live(run)
        if not live.is_empty():
            st.subheader(f"Live feed ({live.height} advisories)")
            st.dataframe(live.reverse().head(200), width="stretch", height=300)
    live_feed()
    adv = df.filter(pl.col("advice") != 0).sort("win", descending=True)
    st.subheader(f"Advisories up to {now:%a %d.%m %H:%M} ({adv.height})")
    per_hour = (adv.with_columns(hour=pl.col("window_start").dt.truncate("1h"),
                                 kind=pl.when(pl.col("advice") == 1).then(pl.lit("BUY")).otherwise(pl.lit("SELL")))
                .group_by(["hour", "kind"]).len().sort("hour"))
    if not per_hour.is_empty():
        f2 = go.Figure()
        for kind, color in (("BUY", GREEN), ("SELL", RED)):
            p = per_hour.filter(pl.col("kind") == kind)
            f2.add_trace(go.Bar(x=p["hour"], y=p["len"], name=kind, marker_color=color))
        f2.update_layout(barmode="stack", height=260, margin=dict(l=10, r=10, t=10, b=10))
        f2.update_xaxes(range=[df_all["window_start"].min(), df_all["window_start"].max()])
        f2.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[22, 7], pattern="hour")])
        st.plotly_chart(f2, width="stretch")
    st.dataframe(adv.with_columns(advice=pl.when(pl.col("advice") == 1).then(pl.lit("BUY")).otherwise(pl.lit("SELL")))
                 .select(["window_start", "symbol", "sectype", "advice", "close", "ema38", "ema100"]),
                 width="stretch")

# ------------------------------------------------------------------ run metrics
with tab_metrics:
    mp = run / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", f"{m['rows']:,}")
        c2.metric("Throughput (events/s)", f"{m['throughput_rows_per_s']:,.0f}")
        lat = m.get("window_close_latency_s", {})
        c3.metric("Window latency p50 / p99", f"{lat.get('p50', 0)*1000:.0f} / {lat.get('p99', 0)*1000:.0f} ms")
        c4.metric("Load imbalance (max/mean)", f"{m['load_imbalance']:.2f}")
        share = m["worker_event_share"]
        st.plotly_chart(go.Figure(go.Bar(x=[f"worker {i}" for i in range(len(share))], y=share))
                        .update_layout(height=250, title="Share of price events per worker",
                                       margin=dict(l=10, r=10, t=40, b=10)), width="stretch")
        cpu = run / "cpu_timeline.csv"
        if cpu.exists():
            c = pl.read_csv(cpu)
            st.plotly_chart(go.Figure([go.Scatter(x=c["t"], y=c["process_tree_pct"], name="engine processes"),
                                       go.Scatter(x=c["t"], y=c["system_pct"], name="system (normalised)")])
                            .update_layout(height=250, title="CPU utilisation (%; 100% = one core for engine)",
                                           margin=dict(l=10, r=10, t=40, b=10)), width="stretch")
        with st.expander("Raw metrics.json"):
            st.json(m)
