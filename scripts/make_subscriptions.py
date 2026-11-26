"""Build the default subscription list (D7): top-N most active symbols + random long-tail sample.

Usage: python scripts/make_subscriptions.py [--counts results/profile/price_events_per_symbol.csv]
                                            [--top 25] [--tail 25] [--seed 42] [--out config/subscriptions.txt]
"""
import argparse
import random
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="results/profile/price_events_per_symbol.csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tail", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="config/subscriptions.txt")
    a = ap.parse_args()
    c = pl.read_csv(a.counts).sort("len", descending=True)
    names = c["ID"].to_list()
    top = names[: a.top]
    rest = names[a.top:]
    tail = random.Random(a.seed).sample(rest, min(a.tail, len(rest)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# top {a.top} most active symbols (price events)\n")
        f.write("\n".join(top) + "\n")
        f.write(f"# {a.tail} random symbols from the long tail (seed {a.seed})\n")
        f.write("\n".join(tail) + "\n")
    print(f"wrote {len(top) + len(tail)} symbols to {a.out}")


if __name__ == "__main__":
    main()
