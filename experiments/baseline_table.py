"""Build the Week 12 baseline table from pulled results.

    python experiments/baseline_table.py results > results/baseline_table.md

Expects results/pairN/results/seedN/<cell>/ from the matrix and, for cells
that were rerun, results/pairN/results/rerun-seedN/<cell>/; a rerun cell
replaces the matrix cell of the same name.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

CELLS = {"checkfreq-0-01": "CheckFreq 1%", "checkfreq-0-05": "CheckFreq 5%", "dcp-46": "DCP every 46 steps"}
PAIRS = [("checkfreq-0-01", "checkfreq-0-05"), ("dcp-46", "checkfreq-0-01"), ("dcp-46", "checkfreq-0-05")]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # two-sided t, by degrees of freedom


def cell_dir(root: Path, seed: int, cell: str) -> Path | None:
    for run in (f"rerun-seed{seed}", f"seed{seed}"):
        d = root / f"pair{seed}" / "results" / run / cell
        if (d / "accounting.json").exists() and "finished:" in (d / "cell.txt").read_text():
            return d
    return None


def redone_losses(d: Path) -> tuple[int, int]:
    """(redone steps, redone steps whose loss differs from the first try)."""
    first: dict[int, float] = {}
    redone = bad = 0
    for line in open(d / "train.jsonl"):
        e = json.loads(line)
        if e.get("event") != "loss":
            continue
        if e["step"] in first:
            redone += 1
            bad += abs(e["loss"] - first[e["step"]]) > 1e-3
        else:
            first[e["step"]] = e["loss"]
    return redone, bad


def memory(run: Path) -> tuple[float, float]:
    """(peak host memory used, lowest available), in GB, over the whole run."""
    rows = list(csv.DictReader(open(run / "memory.csv")))
    return (max(int(r["host_used_mb"]) for r in rows) / 1024,
            min(int(r["host_available_mb"]) for r in rows) / 1024)


def ci(diffs: list[float]) -> str:
    m = statistics.fmean(diffs)
    if len(diffs) < 2:
        return f"{m:+.0f}"
    h = T95[len(diffs) - 1] * statistics.stdev(diffs) / len(diffs) ** 0.5
    return f"{m:+.0f} [{m - h:+.0f}, {m + h:+.0f}]"


def main(root: Path) -> None:
    seeds = sorted(int(p.name[4:]) for p in root.glob("pair*"))
    acc: dict[tuple[str, int], dict] = {}
    dirs: dict[tuple[str, int], Path] = {}
    for s in seeds:
        for c in CELLS:
            d = cell_dir(root, s, c)
            if d is not None:
                dirs[c, s] = d
                acc[c, s] = json.loads((d / "accounting.json").read_text())

    print("## Lost GPU-seconds per training hour (one failure per hour, 2-hour runs)\n")
    print("| Policy | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean |")
    print("|---|" + "---|" * (len(seeds) + 1))
    for c, label in CELLS.items():
        vals = [acc[c, s]["lost_s_per_hour"] if (c, s) in acc else None for s in seeds]
        got = [v for v in vals if v is not None]
        mean = f"{statistics.fmean(got):.0f}" if got else "–"
        print(f"| {label} | " + " | ".join("–" if v is None else f"{v:.0f}" for v in vals) + f" | {mean} |")
    print("| Kills in the run | " + " | ".join(str(acc.get(("checkfreq-0-01", s), {}).get("kills", "–")) for s in seeds) + " | |")

    print("\n## Paired differences (first minus second, seconds per hour, mean and 95% interval over seeds)\n")
    for a, b in PAIRS:
        diffs = [acc[a, s]["lost_s_per_hour"] - acc[b, s]["lost_s_per_hour"] for s in seeds if (a, s) in acc and (b, s) in acc]
        if diffs:
            print(f"- {CELLS[a]} minus {CELLS[b]}: {ci(diffs)} (n = {len(diffs)})")

    print("\n## Parts of the lost time (seconds, summed over seeds)\n")
    print("| Policy | Stall | Slowdown | Redone | Restart | Residual | Saves |")
    print("|---|---|---|---|---|---|---|")
    for c, label in CELLS.items():
        rows = [acc[c, s] for s in seeds if (c, s) in acc]
        if rows:
            f = lambda k: sum(r[k] for r in rows)  # noqa: E731
            print(f"| {label} | {f('stall_s'):.0f} | {f('slowdown_s'):.0f} | {f('redone_s'):.0f} | "
                  f"{f('restart_s'):.0f} | {f('residual_s'):+.1f} | {f('saves'):.0f} |")
    worst = max((abs(a["residual_s"]) for a in acc.values()), default=0.0)
    print(f"\nLargest residual in any one run: {worst:.1f} s.")

    redone = sum(redone_losses(d)[0] for d in dirs.values())
    bad = sum(redone_losses(d)[1] for d in dirs.values())
    print(f"Redone steps that reproduced their logged loss: {redone - bad} of {redone}.")

    print("\n## Host memory (GB of 30)\n")
    for s in seeds:
        for run in (f"seed{s}", f"rerun-seed{s}"):
            p = root / f"pair{s}" / "results" / run
            if (p / "memory.csv").exists():
                used, avail = memory(p)
                print(f"- {run}: peak used {used:.1f}, lowest available {avail:.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
