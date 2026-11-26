"""Knowledge about the DEBS 2022 CSV layout, shared by all components."""
import polars as pl

# Line 12 of each file is the header; lines 1-11 and 13 start with '#'.
COMMENT_PREFIX = "#"

# Columns relevant to the queries (starred in Table 1 of the project description),
# plus the system arrival Date/Time used for profiling and latency analysis.
COL_ID = "ID"
COL_SECTYPE = "SecType"
COL_DATE = "Date"
COL_TIME = "Time"
COL_LAST = "Last"
COL_TRADING_TIME = "Trading time"
COL_TRADING_DATE = "Trading date"

USED_COLUMNS = [COL_ID, COL_SECTYPE, COL_DATE, COL_TIME, COL_LAST, COL_TRADING_TIME, COL_TRADING_DATE]


def read_csv_lazy(path: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazy polars scan of one day file, all used columns as strings."""
    cols = columns or USED_COLUMNS
    return pl.scan_csv(
        path,
        comment_prefix=COMMENT_PREFIX,
        infer_schema=False,  # everything as Utf8; parse explicitly
        truncate_ragged_lines=True,
        ignore_errors=True,
    ).select(cols)
