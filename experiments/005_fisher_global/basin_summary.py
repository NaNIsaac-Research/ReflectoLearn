"""One-file summary of the full basin run, and the way back to the per-candidate files.

The full run writes one JSON per candidate to docs/fisher_global/basin_full/ (935 files, 31.6 MB),
most of it the per-stage optimizer trace `stages`, which basin_report.py does not read. The files
are kept as the release asset `basin_full.zip`; the repository keeps basin_full_summary.jsonl, every
field except `stages`, one candidate per line, sorted by file name.

    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin_summary.py pack
    .venv/Scripts/python.exe -I experiments/005_fisher_global/basin_summary.py unpack

`pack` reads basin_full/ and writes the summary. `unpack` writes basin_full/ back from the summary
(without `stages`), which is enough for basin_report.py to rebuild basin_full.md and
basin_verdict.json. For the traces, unzip the release asset into docs/fisher_global/ instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "fisher_global"
FULL_DIR = OUT / "basin_full"
SUMMARY = OUT / "basin_full_summary.jsonl"
DROPPED = ("stages",)


def pack() -> None:
    with SUMMARY.open("w", encoding="utf-8", newline="\n") as out:
        for path in sorted(FULL_DIR.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            for key in DROPPED:
                row.pop(key, None)
            out.write(json.dumps({"file": path.name, **row}) + "\n")


def unpack() -> None:
    FULL_DIR.mkdir(exist_ok=True)
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        (FULL_DIR / row.pop("file")).write_text(json.dumps(row), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["pack", "unpack"])
    {"pack": pack, "unpack": unpack}[ap.parse_args().command]()


if __name__ == "__main__":
    main()
