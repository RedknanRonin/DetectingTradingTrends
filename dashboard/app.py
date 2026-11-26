"""Trading-trend dashboard (D8).  Run:  streamlit run dashboard/app.py

Views
  * Signal board  - one tile per subscribed symbol: current trend (EMA38 vs EMA100), the
                    EMA spread in %, and the latest advisory, sorted so that fresh
                    crossovers stand out ("cut through the noise").
  * Symbol detail - Close per window with EMA38/EMA100, BUY/SELL markers, shaded
                    bullish/bearish regimes, and the EMA spread panel.
  * Advisory feed - newest advisories first; live-tails advisories_live.jsonl while a run
                    is in progress.
  * Run metrics   - throughput, latency, CPU and per-worker load of the selected run.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
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
        tiles.append(
            f'<div style="border-left:6px solid {GREEN if bull else RED};padding:6px 10px;'
            f'border-radius:4px;background:rgba(127,127,127,0.08)">'
            f'<b>{r["symbol"]}</b> <small>{"Index" if r["sectype"] == "I" else "Equity"}</small><br>'
            f'<span style="font-size:1.25em">{r["close"]:.4g}</span>'
            f'<span style="color:{GREEN if bull else RED}">&nbsp;{r["spread_pct"]:+.2f}%</span><br>'
            f'<small style="color:{adv_col}">{adv} {when}</small></div>')
    st.markdown('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));'
                'gap:10px">' + "".join(tiles) + "</div>", unsafe_allow_html=True)
    st.caption(f"Showing {shown.height} of {board.height} symbols. Border colour = current regime "
               "(green: EMA38 above EMA100). Percentage = EMA spread (EMA38 &minus; EMA100)/EMA100.")

# ------------------------------------------------------------------ symbol detail
with tab_symbol:
    syms = board.sort("windows", descending=True)["symbol"].to_list()  # most data first
    sym = st.selectbox("Symbol (sorted by number of windows)", syms)
    d = df.filter(pl.col("symbol") == sym).sort("win")
    x = d["window_start"].to_list()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)
    # shade bullish / bearish regimes
    bull = (d["ema38"] > d["ema100"]).to_numpy()
    if len(bull):
        starts = np.flatnonzero(np.r_[True, bull[1:] != bull[:-1]])
        ends = np.r_[starts[1:], len(bull)] - 1
        for s, e in zip(starts, ends):
            fig.add_vrect(x0=x[s], x1=x[e], fillcolor=GREEN if bull[s] else RED, opacity=0.06,
                          line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["close"], name="Close (last price per window)",
                             mode="lines", line=dict(color=GREY, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema38"], name="EMA38", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["ema100"], name="EMA100", line=dict(color=ORANGE, width=2)), row=1, col=1)
    for code, name, color, symbol in ((1, "BUY", GREEN, "triangle-up"), (-1, "SELL", RED, "triangle-down")):
        a = d.filter(pl.col("advice") == code)
        fig.add_trace(go.Scatter(x=a["window_start"], y=a["close"], mode="markers", name=name,
                                 marker=dict(color=color, size=13, symbol=symbol,
                                             line=dict(color="white", width=1))), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=d["spread_pct"], name="EMA spread %",
                         marker_color=np.where(bull, GREEN, RED)), row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04))
    # hide nights and weekends (no trading) so the trading sessions sit next to each other
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[22, 7], pattern="hour")])
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="spread %", row=2, col=1)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(d.filter(pl.col("advice") != 0).select(
        ["window_start", "advice", "close", "ema38", "ema100", "prev38", "prev100"]),
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
    st.subheader(f"All advisories in run ({adv.height})")
    per_hour = (adv.with_columns(hour=pl.col("window_start").dt.truncate("1h"),
                                 kind=pl.when(pl.col("advice") == 1).then(pl.lit("BUY")).otherwise(pl.lit("SELL")))
                .group_by(["hour", "kind"]).len().sort("hour"))
    if not per_hour.is_empty():
        f2 = go.Figure()
        for kind, color in (("BUY", GREEN), ("SELL", RED)):
            p = per_hour.filter(pl.col("kind") == kind)
            f2.add_trace(go.Bar(x=p["hour"], y=p["len"], name=kind, marker_color=color))
        f2.update_layout(barmode="stack", height=260, margin=dict(l=10, r=10, t=10, b=10))
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
