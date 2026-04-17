"""
Offline replay to reproduce Run B's features.parquet from raw NDJSON.

Uses the SAME ProductState.ingest() logic as the live Kafka featurizer
(features/featurizer.py) so the output is semantically identical to what
Run B trained on. The only difference is this script reads NDJSON directly
instead of consuming from Kafka.

Progress is printed to stderr with flush=True every N ticks so we can
monitor long-running replays.
"""

# Standard library imports — json for line parsing, pathlib for paths, sys for stderr, time for profiling
import json
import pathlib
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Import the production ProductState + schema from features/featurizer.py
# so the replay is byte-compatible with what Run B's live featurizer produced
sys.path.insert(0, "features")
from featurizer import ProductState, PARQUET_SCHEMA

# Run B config values (copied verbatim from config.yaml so we don't accidentally drift)
WINDOW_SEC = 60.0
HORIZON_SEC = 60.0
VOL_THRESHOLD = 0.000048

# Run B collection end — model card says "~65 hours", last timestamp 2026-04-07 15:54:58
RUN_B_END_TS = "2026-04-07T15:54:58"

# Raw NDJSON files in chronological order — the same sequence Kafka would have replayed
RAW_DIR = pathlib.Path("/Users/ricopichardo/Documents/BitcoinProjectTest/data/raw")
FILES = [
    "20260404.ndjson",
    "20260405.ndjson",
    "20260406.ndjson",
    "BTC-USD_20260406.ndjson",
    "BTC-USD_20260407.ndjson",
]

# Output parquet — keep .runB suffix so we don't clobber Run C's data/processed/features.parquet
OUT_PATH = pathlib.Path("data/processed/features.runB.parquet")


def log(msg: str) -> None:
    # Print to stderr with flush=True so progress is visible immediately (stdout is buffered)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    # Single ProductState covers all BTC-USD ticks across files
    state = ProductState(WINDOW_SEC, HORIZON_SEC, VOL_THRESHOLD)
    rows: list[dict] = []

    t0 = time.time()
    total_ticks = 0

    # The cutoff prefix below is padded so any nanosecond timestamp like
    # "2026-04-07T15:54:58.900000000Z" still compares <= to the cutoff string
    cutoff_upper = RUN_B_END_TS + ".999999999Z"

    for name in FILES:
        fp = RAW_DIR / name
        file_ticks = 0
        file_start = time.time()
        with open(fp) as f:
            for line in f:
                try:
                    tick = json.loads(line)
                except Exception:
                    # Malformed NDJSON line — skip silently, matches live featurizer behaviour
                    continue
                if tick.get("product_id") != "BTC-USD":
                    # Defensive: raw files should only contain BTC-USD but check anyway
                    continue
                # Early cutoff — stop ingesting ticks past Run B's collection end
                if tick["timestamp"] > cutoff_upper:
                    continue
                try:
                    labelled = state.ingest(tick)
                except Exception:
                    # Same graceful skip as the live featurizer's KeyError/ValueError handler
                    continue
                rows.extend(labelled)
                total_ticks += 1
                file_ticks += 1

                # Print progress every 25k ticks so we can watch long replays
                if total_ticks % 25_000 == 0:
                    rate = total_ticks / (time.time() - t0)
                    log(
                        f"  processed {total_ticks:,} ticks  "
                        f"({rate:.0f}/s)  labelled rows={len(rows):,}"
                    )
        dt = time.time() - file_start
        log(
            f"  finished {name}  (+{file_ticks:,} ticks in {dt:.1f}s, "
            f"total rows so far={len(rows):,})"
        )

    # Drain any pending rows whose lookahead window has closed by end-of-stream
    rows.extend(state.drain_remaining())
    log(f"Total labelled rows after drain: {len(rows):,}")

    # Belt-and-suspenders: trim rows whose row-timestamp sits past Run B cutoff
    rows = [r for r in rows if r["timestamp"] <= cutoff_upper]
    log(f"After Run B cutoff trim: {len(rows):,}")

    # Write to parquet using the production schema so downstream code treats
    # this identically to the live-generated features.parquet
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(table, OUT_PATH)

    # Summary — compare against metadata.json's 613,853 target
    df = pd.read_parquet(OUT_PATH)
    print()
    print("=== reproduced Run B features.parquet ===")
    print(f"  rows:        {len(df):,}  (Run B metadata: 613,853)")
    print(f"  time range:  {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  spike rate:  {df['vol_spike'].mean():.4f}  (model card: ~0.135)")
    print(f"  runtime:     {time.time() - t0:.1f}s")
    print(f"  output:      {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
