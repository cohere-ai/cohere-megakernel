#!/usr/bin/env python3
"""Parse runner benchmark dumps into markdown tables.

Usage: first run the benchmark 
```bash
function run_benchmark() {
    local INLEN=$1
    local BS=$2
    local OUTLEN=$3
    echo "running benchmark with INLEN=$INLEN, BS=$BS, OUTLEN=$OUTLEN"
    python src/runner.py --mk --lib-path "$PWD/build/libmk_release.so" \
    --checkpoint <PATH_TO_CHECKPOINT> \
    --batch-size $BS --max-new-tokens $OUTLEN --num-sms 132 --frac-vram-utilization 0.95 --cpp-decode-runtime \
     --fake-prompt-len $INLEN --fast --real-weight  | grep "metrics"
}
run_benchmark 8192 1 1024 >> benchmark.txt
run_benchmark 8192 4 1024 >> benchmark.txt
```
then run the script
```bash
python tools/print_perf_table.py benchmark.txt
```

Input lines look like::

    running benchmark with INLEN=64, BS=1, OUTLEN=128
    [mk-release] metrics = {'tpot_ms': 3.76, 'throughput_tok_s': 265.4, ...}

Prints Mean TPOT (ms) and Decode throughput (tok/s) grids keyed by (BS, INLEN).
Missing cells are ``N/A``. Values come straight from the dumped metrics dict.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from typing import Iterable


RUN_MARKER_RE = re.compile(
    r"^running benchmark with INLEN=(?P<inlen>\d+),\s*"
    r"BS=(?P<bs>\d+),\s*OUTLEN=(?P<outlen>\d+)\s*$"
)
METRICS_RE = re.compile(r"^\[mk-release\]\s+metrics\s*=\s*(?P<body>\{.*\})\s*$")


@dataclass(frozen=True)
class RunRecord:
    inlen: int
    bs: int
    tpot_ms: float
    throughput_tok_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse runner benchmark dumps "
            "and print Mean TPOT + decode throughput markdown tables."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="log files to parse; reads stdin when omitted",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="decimal places for numeric cells (default: 2)",
    )
    return parser.parse_args()


def iter_input_lines(paths: list[str]) -> Iterable[str]:
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            yield from fh


def parse_runs(lines: Iterable[str]) -> list[RunRecord]:
    records: list[RunRecord] = []
    pending_inlen: int | None = None
    pending_bs: int | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        marker = RUN_MARKER_RE.match(line)
        if marker is not None:
            pending_inlen = int(marker.group("inlen"))
            pending_bs = int(marker.group("bs"))
            continue

        metrics_match = METRICS_RE.match(line)
        if metrics_match is None:
            continue
        if pending_inlen is None or pending_bs is None:
            # Orphan metrics line (no preceding run marker). Skip rather than
            # inventing a (bs, inlen) key.
            continue

        metrics = ast.literal_eval(metrics_match.group("body"))
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics payload is not a dict: {metrics!r}")
        if "tpot_ms" not in metrics or "throughput_tok_s" not in metrics:
            raise ValueError(
                "metrics dict missing tpot_ms / throughput_tok_s; "
                f"keys={sorted(metrics)}"
            )

        records.append(
            RunRecord(
                inlen=pending_inlen,
                bs=pending_bs,
                tpot_ms=float(metrics["tpot_ms"]),
                throughput_tok_s=float(metrics["throughput_tok_s"]),
            )
        )
        pending_inlen = None
        pending_bs = None

    return records


def _cell_width(inlens: list[int], decimals: int) -> int:
    # Match the example layout: right-align numbers under the Input Len header.
    # Width is max(header digits, formatted value width for e.g. 99999.99).
    header_w = max(len(str(v)) for v in inlens)
    value_w = decimals + 1 + 5  # e.g. 99999.99
    return max(header_w, value_w)


def render_table(
    *,
    title: str,
    records: list[RunRecord],
    value_of,
    decimals: int,
) -> str:
    if not records:
        return f"### {title}\n\nNo complete runs found in input."

    inlens = sorted({r.inlen for r in records})
    bs_values = sorted({r.bs for r in records})
    by_key = {(r.bs, r.inlen): value_of(r) for r in records}

    col_w = _cell_width(inlens, decimals)
    # Left label column: "BS \ Input Len" is longer than any BS integer.
    label = "BS \\ Input Len"
    label_w = max(len(label), max(len(str(bs)) for bs in bs_values))

    fmt = f"{{:>{col_w}.{decimals}f}}"
    na = f"{{:>{col_w}}}".format("N/A")

    header_cells = [f"{label:>{label_w}}"] + [f"{n:>{col_w}d}" for n in inlens]
    sep = "| " + " | ".join(["---"] * (len(inlens) + 1)) + " |"

    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(header_cells) + " |",
        sep,
    ]
    for bs in bs_values:
        row = [f"{bs:>{label_w}d}"]
        for inlen in inlens:
            value = by_key.get((bs, inlen))
            row.append(na if value is None else fmt.format(value))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.decimals < 0:
        raise SystemExit("--decimals must be non-negative")
    records = parse_runs(iter_input_lines(args.inputs))
    # Last write wins if a (bs, inlen) pair appears more than once in the dump.
    dedup: dict[tuple[int, int], RunRecord] = {}
    for record in records:
        dedup[(record.bs, record.inlen)] = record
    records = list(dedup.values())

    print(
        render_table(
            title="Mean TPOT (ms)",
            records=records,
            value_of=lambda r: r.tpot_ms,
            decimals=args.decimals,
        )
    )
    print()
    print(
        render_table(
            title="Decode throughput (tok/s)",
            records=records,
            value_of=lambda r: r.throughput_tok_s,
            decimals=args.decimals,
        )
    )


if __name__ == "__main__":
    main()
