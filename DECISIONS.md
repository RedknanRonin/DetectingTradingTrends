# Architecture & Design Decisions

Decision log for the CS-E4780 course project (DEBS 2022 GC: Detecting Trading Trends).
Decisions are made by the student; this file records each choice, the alternatives
considered, and the rationale, so the report can reference them.

Hardware: Intel i7-12700H (6P + 8E cores = 14 cores / 20 threads), 15.7 GB RAM, Windows 11.

---

## D1. Processing platform: custom Python multi-process stream processor
- **Chosen:** own engine in Python: source → hash-partition by symbol → N worker processes
  holding per-symbol state → sink.
- **Alternatives:** Apache Flink (Docker), custom Rust, Kafka + Flink.
- **Rationale:** every mechanism (partitioning, windowing, watermarks, state) is explicit
  and explainable; core-scaling experiments are straightforward; low setup risk given the deadline.
- **Known trade-off:** Python parsing/IPC overhead; the single source may become the
  bottleneck. This is a measurable and reportable limitation.

## D2. Dataset: full week (Mon 8.11 – Sun 14.11.2021), ~25 GB CSV
- Stored outside OneDrive in `C:\data\debs2022` (to avoid syncing 25 GB); path is configurable.
- Day files: 08 (4.67 GB), 09 (5.06), 10 (5.51), 11 (4.79), 12 (4.84), 13 and 14 (~0 GB, weekend).

## D3. Ingestion: replay CSV files as fast as possible
- Events are streamed in file order at maximum speed; windows use event time (D4).
- Measures maximum sustainable throughput; latency is measured as processing delay.
- **Alternatives:** paced real-time replay (infeasible for a full week), message broker (extra infrastructure).

## D4. Time semantics: event time = Trading date + Trading time
- Windows: 5-minute tumbling, aligned to 00:00 CEST each day (w0 = 00:00–00:05).
- **Alternative:** system arrival Date/Time (≈ processing time, fewer ordering issues, but
  not the trading meaning of the window).

## D5. Late / out-of-order events: watermark + bounded lateness
- Watermark = max observed event time − allowed lateness (configurable, e.g. a few seconds).
- Window w_i of all symbols is evaluated once the watermark passes the end of w_i
  (i.e. "w_i is evaluated when w_{i+1} starts", in event time).
- Events arriving for an already-closed window are counted as *late* and dropped (reported).
- **Alternatives:** strict close on first later event (lowest latency, more drops);
  per-symbol closing (no global clock, but illiquid symbols emit very late).
- **Experiment:** vary lateness → trade-off between latency and number of dropped events.

## D6. Windows without price events: skip (EMA unchanged)
- Close_{s,w} exists only if ≥1 price (`Last`) event for s in w. Otherwise no EMA update
  and no crossover check; state carries over to the next window with a price.
- **Alternative:** carry the last Close forward and update all symbols every window.
- **Rationale:** matches the definition "Close = last price event observed in w_i";
  only active symbols cost work, which helps with the long-tail distribution.

## D7. Subscriptions: configurable list with a default set
- Metrics (EMA38, EMA100, crossovers) are computed for **all** symbols; output is emitted
  only for subscribed symbols.
- Default subscription: top-N most active symbols + a sample of long-tail symbols; can be
  overridden by config/CLI. Subscription size can be varied as a scalability experiment.

## D8. Visualization: interactive web dashboard
- Per-symbol chart: price (Close per window), EMA38, EMA100, with buy/sell markers at crossovers.
- Advisory feed and an overview (most active symbols, most crossovers).

## D9. Partitioning: hash partitioning + load-balanced variant (compared)
- **Default:** worker = hash(symbol) mod N (keyBy-style, stateless routing).
- **Variant:** greedy load-balanced assignment: symbols sorted by expected event count
  (from a profiling pass / previous day), heaviest first to the least-loaded worker.
- **Experiment:** compare per-worker event counts (skew), throughput and latency of both
  under the long-tail distribution.
- **Not chosen:** dynamic rebalancing with state migration (too risky for the timeline).

## D10. Ingestion/parsing: single source process, columnar chunks
- Source reads CSV in large chunks with pyarrow, only the relevant columns
  (ID, SecType, Last, Trading time, Trading date); events without a price are filtered
  early (counted, and still used to advance event time where they have a timestamp).
- Sends compact numpy batches per partition to the workers.
- **Alternatives:** separate parallel normalization stage to Parquet; parallel parser pool.
- **Known limitation:** the single source caps scalability; measure and explain (Amdahl).

## D11. No advisory in a symbol's first window (+ optional warm-up)
- **Problem:** read literally, the formulas emit a BUY for every symbol in its first window:
  with EMA_0 = 0, the first window gives EMA38 = 2/39·Close > EMA100 = 2/101·Close while both
  previous values are 0 (0 ≤ 0), so Eq. (1)–(2) hold. This is a fake crossover caused by the
  initialisation, not a price movement (4,906 such BUYs on Tue+Wed, one per symbol).
- **Chosen (user decision):** the crossover check is skipped in a symbol's first window, since
  there is no real previous EMA to cross from. This is the only window where the artefact
  occurs. EMA values themselves follow the spec exactly (EMA_0 = 0). Implemented identically in
  the operator and the reference (`eligible = windows_before >= max(1, K)`).
- **Optional warm-up K** (`--warmup`): additionally require K previous windows before any
  advisory; K = 0 or 1 now means "first-window rule only". Default currently K = 38.
- **Alternatives considered:** seed both EMAs with the first Close (removes the artefact and
  the hours-long EMA start-up, but changes the EMA values defined by the spec); rely on the
  warm-up only (artefact still emitted when K = 0); follow the spec literally.

## D12. Correctness verification: independent reference implementation + unit tests
- A simple batch implementation (polars) computes per-window Close, EMA38/EMA100 and
  crossovers from the same data; the streaming output is diffed against it, with
  differences explained (e.g. late events dropped by the watermark).
- Unit tests on hand-crafted event sequences (EMA math, window boundaries, crossovers,
  late events, empty windows, warm-up).

## Implementation defaults (proposed, can be overridden)
- **Core-count experiments:** restrict the whole process tree to k logical CPUs via CPU
  affinity (psutil), k ∈ {1, 2, 4, 8, 14}; workers = k−1 (source takes one).
  Note: P-cores vs E-cores differ in speed; affinity pins to P-cores first.
- **Metrics:** throughput (events/s, total and per stage), window-close latency
  (wall time from watermark passing the window end → result emitted), per-worker load,
  CPU utilization (psutil sampling), late-event count.
- **Outputs:** Parquet/CSV files per run: window results (subscribed symbols), advisories,
  metrics; consumed by the dashboard and the evaluation scripts.
- **Dashboard:** Streamlit + Plotly reading the output files.
- **Libraries:** pyarrow, numpy, psutil, polars (reference), streamlit, plotly, pytest.

---

## Query semantics (fixed by the spec)
- EMA^j_{s,w_i} = Close_{s,w_i} · 2/(1+j) + EMA^j_{s,w_{i-1}} · (1 − 2/(1+j)), j ∈ {38, 100}, EMA_{s,w0} = 0.
- Bullish (BUY): EMA38_i > EMA100_i and EMA38_{i-1} ≤ EMA100_{i-1}.
- Bearish (SELL): EMA38_i < EMA100_i and EMA38_{i-1} ≥ EMA100_{i-1}.
- Symbol = ID.[Exchange] (e.g. RDSA.NL); all events grouped by symbol.

## Data findings (profiling, `scripts/profile_data.py`) and resulting rules
Numbers from the first ~13M rows of Monday; re-run on the full week for the report.
- **File format:** 11 comment lines, header, a `#'...'` description line; data rows have a
  trailing comma (40 fields vs 39 header names). Empty fields must be read as null.
- **Price events:** `Last > 0` and `Trading time` present; ~21% of all events. `Last = 0`
  (with `Trading time 00:00:00`) are placeholder/initialisation records → treated as non-price.
- **Event date:** `Trading date` is filled for only ~9% of price events (and then always equals
  `Date`) → event time = coalesce(`Trading date`, `Date`) + `Trading time`. (Normalisation rule.)
- **System `Time` is truncated to whole seconds** (99.7% end in `.000`), so it cannot be used
  for sub-second ordering; Trading time has ms precision.
- **Ordering:** per symbol, event time never decreases in file order. Globally, events are up to
  ~1 s out of order (max 956 ms) → lateness 1000 ms ⇒ zero drops; lateness 0 ⇒ a few drops at
  window boundaries.
- **Long tail:** top 100 symbols ≈ 22% of price events; top 1000 ≈ 65%.
- **EMA_0 = 0 effect:** EMA38 approaches the price much faster than EMA100, so EMA38 > EMA100
  for hours after a symbol starts; in a 4-h test, the *only* advisories with K = 0 were the
  50 first-window BUYs (one per subscribed symbol). Real crossovers only appear after EMA100 has
  converged (time constant ≈ 50 windows ≈ 4 h of trading).

## Implementation decisions (made during implementation)
- **I1. Late filtering at the source, row by row.** The watermark is global and the source
  sees all events in order, so it marks an event late iff its window was already closed by the
  watermark *before that row*. Equivalent to event-at-a-time semantics and independent of the
  block size. The operator additionally drops events for already-evaluated windows (safety net).
- **I2. One message per block to every worker** (its partition + the watermark), so all
  workers advance event time together even when their partition slice is empty.
- **I3. Vectorised keyed state.** Per-symbol state (EMA38, EMA100, window count) lives in numpy
  arrays indexed by a global integer symbol id; a block is reduced to the last price per
  (symbol, window) with a stable sort, and window evaluation is vectorised per window.
  Cost scales with the number of *active* (symbol, window) pairs, not with all symbols.
- **I4. Bounded queues** (default 8 blocks per worker) → backpressure instead of unbounded memory.
- **I5. Malformed rows are skipped and counted** (pyarrow invalid-row handler).
- **I6. Preload mode** (`--preload`): parse everything first, then stream from memory. Isolates
  the scalability of the partitioned processing stage from the single-source parsing bottleneck.
- **I7. Latency metrics:** window-evaluation latency (source starts reading the block whose
  watermark closes a window → EMAs/crossovers computed at the worker) for all symbols; sink
  latency for emitted results. Throughput measured from stream start to last result.
- **I8. Core pinning order** on the i7-12700H: one hyper-thread per P-core first (CPUs 0,2,..,10),
  then the 8 E-cores (12–19), then the remaining P-core hyper-threads.
- **I9. Benchmark validity:** runs while downloading / with a cold page cache were ~3× slower;
  benchmarks use a discarded warm-up run. One weekday (~5 GB) fits in the OS page cache; the
  full week (25 GB) does not (16 GB RAM) → full-week runs include disk reads.

- **I10. 1-minute OHLC bars for subscribed symbols (dashboard, user decision).** Workers buffer
  the price events of *subscribed* symbols only and emit open/high/low/close/count per
  (symbol, minute) once the watermark passes the minute's end (same mechanism as windows).
  Visualisation only, not part of Q1/Q2. Consistency check: every window Close equals the close
  of its last minute bar (2,248/2,248 on partial Monday). Throughput impact not measurable
  (~605k rows/s with bars). `--no-bars` disables it. Alternatives considered: keep 5-minute
  granularity only; store raw ticks of subscribed symbols (larger output).
- **I11. Dashboard replay.** A time cursor ("now"); every view shows only what was known at
  that time (windows with end <= now, bars with end <= now). Step = 1 minute or 5 minutes.
  Window values are plotted at the window *end* (when they become known), aligned with candles.
- **I12. Per-symbol activity and tiers.** Workers count price events per symbol (keyed state);
  the sink writes `symbols.parquet` (symbol, SecType, price events, windows, owning worker) for
  *all* symbols; the counts reconcile exactly with the source's price-event total. The dashboard
  groups symbols into rank-based activity tiers over all symbols of the run: very high (top 1%),
  high (1-10%), medium (10-50%), low (bottom 50% = long tail). Same table gives per-worker load.
- **I13. EMA visibility in the price-zoomed chart.** Because EMA_0 = 0, EMAs sit far below the
  price until they converge; the price-zoomed y-axis therefore includes only converged EMA
  points (within 25% of Close) and states how many windows' EMAs are out of view.
- **I14. Invalid (stale) trading timestamps are dropped, in stream AND reference (user decision).**
  Found by the Tue+Wed validation: one burst on Wed 10.11 at 18:56:11–13 (system time) of
  5,506 *index* price events from 1,812 symbols stamped `Trading time = 01:00:00.000`
  (≈18 h behind arrival; several different prices per symbol within the same second, i.e. a
  replay, not live prices). The stream dropped them as late (correct under D5), while the
  reference put them into a Wed 01:00 window, inserting a spurious window before Wednesday's
  open → 1,812 missing rows, EMA deviations up to 49%, ~60 differing advisories.
  Rule: a price event whose Trading time is more than 5 min *behind* its arrival time
  (system Date + Time) is an invalid stamp: dropped and counted as `stale_timestamp_dropped`
  (not as late), before it can affect the watermark. Same rule in `scripts/reference.py`.
  No other day so far has any price event more than 1 s behind arrival.
  Alternatives considered: repair with arrival time (would inject replayed prices as new
  Closes after market close); keep and explain the validation mismatch.
- **Not an anomaly: Xetra closing auction.** ~500 ETR equities per day (Tue, Wed) have
  `Trading time` 17:30:xx while their system `Time` is earlier (up to 8.5 h). Their file
  position matches ~17:30 and 17:30 is Xetra's closing auction, so the trading time is right
  and `Time` is the record's previous update. They do not advance the watermark beyond real
  time and are untouched by I14 (which only checks timestamps far *behind* arrival).
- **Measured event rates** (partial Monday, 09:00–11:30, all exchanges open): all events
  mean 1,850/s (p99 5,500, max 10,285); price events mean 343/s (p99 2,545, max 3,106);
  price events per (symbol, 5-min window): median 20, p99 300, max 1,610.

- **I15. Live mode: real-time socket feed (user request, branch `live-socket`).** Extension of
  D3: besides replaying files as fast as possible, the data can be fed at its real arrival rate.
  * `scripts/feed_server.py` is a TCP server that plays the day files back at their arrival
    rate (system Date + Time), xS faster if wanted: plain text, the CSV header first, then one
    CSV line per event in file order (all three exchanges interleaved, as in the files).
    `--fast-until` sends the morning without pacing (EMA history), `--until` stops the feed.
  * The feeder paces on a robust clock, not the raw stamps: file order is the arrival order,
    but some rows have bogus Date+Time (00:00:00 placeholders, the opening snapshot including
    several hundred rows with the previous evening's ~17:35 time, scattered far-future values,
    the Xetra closing-auction rows above). Measured on Mon/Wed/Fri, far-ahead rows occur in runs
    of at most ~540 rows, so the clock = running max of the median of the last ~4000 stamps
    (sampled every 8th row): at most ~5 min ahead (briefly, at the file start), typically ~1 s
    behind true time, so the feed runs ~1 s late, consistently.
  * `SocketSource` (`--socket HOST:PORT`) reuses the file source's transform (filters, event
    time, late filtering, routing); only batching differs: a batch is cut every `--flush-ms`
    (default 100) or when a block is full, so a slow feed is not held back waiting for 16 MB.
  * Idle watermark: without data the watermark would never move (event-time only), so after
    `--idle-ms` (default 2000) without data, event time advances with the wall clock x
    `--clock-rate` (the feeder's speed). Open windows then still close when the market is quiet.
  * Latency metrics start when the batch's first byte arrived, so they include the flush delay.
  * Checked: Wed 00:00-10:00 (09:00-10:00 at 60x, 8.45M rows) with `--emit-all` gives the same
    64,376 windows as the reference on the same rows (EMA rel. diff <= 7e-15); 0 late events.
  * Not done: fault tolerance/checkpointing; the dashboard still reads the outputs written at
    the end of a run (advisories are written live to `advisories_live.jsonl` and printed).

## Open questions
- Default K for the warm-up (currently 38) once full-week crossover counts are known.
