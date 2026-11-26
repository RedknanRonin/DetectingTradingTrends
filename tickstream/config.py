"""Run configuration (all knobs used in the experiments)."""
from dataclasses import dataclass, field

WINDOW_MS = 5 * 60 * 1000  # 5-minute tumbling windows, aligned to 00:00 (D4)
EMA_J = (38, 100)          # smoothing factors j1 < j2


@dataclass
class RunConfig:
    files: list[str]
    out_dir: str = "results/runs/default"
    workers: int = 2                 # number of worker (window/EMA operator) processes
    cores: list[int] | None = None   # logical CPUs to pin the whole run to (None = all)
    partitioning: str = "hash"       # "hash" | "balanced"   (D9)
    weights_file: str | None = None  # per-symbol event counts for balanced partitioning
    lateness_ms: int = 1000          # allowed lateness of the watermark (D5)
    warmup_windows: int = 38         # K: windows required before advisories (D11); 0 = literal spec
    subscriptions_file: str | None = None  # one symbol per line (D7); None = subscribe to none
    emit_all: bool = False           # emit window results for ALL symbols (validation runs)
    block_size: int = 16 << 20       # CSV bytes per source batch
    queue_size: int = 8              # bounded queues -> backpressure
    max_rows: int | None = None      # stop after this many CSV rows (quick tests)
    source_threads: int | None = None  # pyarrow parser threads (None = len(cores) or all)
    live_advisories: bool = True     # append advisories to a JSONL file while running
    preload: bool = False            # parse all input first, then stream from memory
    extra: dict = field(default_factory=dict)
