# tickstream: Detecting Trading Trends in Financial Tick Data

CS-E4780 Scalable Systems and Data Management, course project 2026.
A partitioned stream processor (Python, multi-process) for the DEBS 2022 Grand Challenge
queries:

- **Query 1:** EMA38 and EMA100 per symbol over 5-minute tumbling event-time windows.
- **Query 2:** BUY/SELL advisories on bullish/bearish EMA crossovers.
- **Dashboard:** an interactive dashboard that shows the results.

Design decisions and their rationale are documented in [DECISIONS.md](DECISIONS.md).

## Architecture

```
CSV day files ──► Source (1 process) ──bounded queues──► Worker 0..N-1 ──► Sink (main process)
                  pyarrow columnar parse                 keyed state per     window results,
                  price filter, event time,              symbol: Close,      advisories (live JSONL),
                  symbol registry, watermark,            EMA38/EMA100,       metrics, CPU sampling
                  late filter, partition routing         crossovers
```

| Module | Role |
|---|---|
| `tickstream/source.py` | CSV parsing, normalization, symbol registry, watermark and late filter, hash/balanced partitioning |
| `tickstream/operator.py` | Window/EMA/crossover operator (pure, unit-tested) |
| `tickstream/engine.py` | Process topology, queues, sink, metrics, CPU pinning |
| `scripts/profile_data.py` | Dataset profiling (event types, long tail, disorder, rates) |
| `scripts/reference.py` / `scripts/validate.py` | Independent batch reference (polars) and diff |
| `scripts/bench.py` | Experiment matrix (cores, preload, partitioning, lateness, block size) |
| `dashboard/app.py` | Streamlit dashboard |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows (Linux/macOS: .venv/bin/pip)
```

Download the dataset (CC BY-NC-SA 4.0) from https://doi.org/10.5281/zenodo.6382482.
The examples below assume it is in `C:/data/debs2022/`.

## Usage

```bash
# profile the data, then build the default subscription list (top 25 + 25 long-tail symbols)
python scripts/profile_data.py C:/data/debs2022/*.csv --out results/profile
python scripts/make_subscriptions.py

# run the full week on 8 cores
python -m tickstream "C:/data/debs2022/debs2022-gc-trading-day-*.csv" --cores 8 --out results/runs/week

# validate against the reference (all symbols)
python -m tickstream "C:/data/debs2022/*.csv" --emit-all --out results/runs/week_all
python scripts/reference.py "C:/data/debs2022/*.csv" --out results/reference_week.parquet
python scripts/validate.py results/runs/week_all/windows.parquet results/reference_week.parquet

# experiments
python scripts/bench.py C:/data/debs2022/debs2022-gc-trading-day-09-11-21.csv --exp cores preload partition --warmup-run --tag tue

# dashboard
streamlit run dashboard/app.py

# tests
python -m pytest tests
```

Important options of `python -m tickstream`:

| Option | Meaning |
|---|---|
| `--cores` | Pin to *k* cores |
| `--workers` | Number of workers (default *k*) |
| `--partitioning hash\|balanced` | How symbols are assigned to workers |
| `--lateness-ms` | Allowed lateness (default 1000) |
| `--warmup K` | Advisory warm-up in windows (default 38; 0 = literal spec) |
| `--emit-all` | Emit results for every symbol, not only subscribed ones |
| `--preload` | Processing-only mode |
| `--block-mb` | Source block size |

Outputs in `--out`:

| File | Contents |
|---|---|
| `windows.parquet` | Per (symbol, window): Close, EMA38, EMA100, previous EMAs, advice |
| `advisories.parquet` | Advisories only |
| `advisories_live.jsonl` | Advisories written while the run is in progress |
| `metrics.json` | Run metrics |
| `cpu_timeline.csv` | CPU utilisation over time |

## Use of AI tools

The implementation was written with Claude Code (an AI coding agent) according to
architecture decisions made by the student (see DECISIONS.md). The student reviewed and
verified the code and results. The report was written by the student.
