"""Lost-time accounting from the execution record.

Lost time is computed from timestamps the policy did not choose:

    lost = (run time after the first step began) - useful steps x clean step time

and split into four parts that should add up to it:

- stall: the training loop blocked by saves that were kept;
- slowdown: extra step time while a background write ran, on kept steps;
- redone: for each kill, the time from the end of the restored step to the
  kill (the age of the newest finished checkpoint at the failure);
- restart: from each kill to the start of the first step after restore.

The residual (lost minus the four parts) is reported, not hidden; a large
residual means the accounting is missing something.

    python -m facp.accounting run/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from facp.record import read


def account(train_record: str | Path, supervisor_record: str | Path,
            clean_step: float | None = None) -> dict:
    sup = list(read(supervisor_record))
    events = list(read(train_record))

    start = next(e for e in sup if e["event"] == "run_start")
    end = next((e for e in sup if e["event"] == "run_end"), None)
    if end is None:
        raise ValueError("run has no run_end event; it did not finish")
    kills = [e for e in sup if e["event"] == "kill"]

    # split the training record into incarnations at each process start
    incarnations: list[list[dict]] = []
    for e in events:
        if e["event"] == "start" or not incarnations:
            incarnations.append([])
        incarnations[-1].append(e)

    times = [e for e in events if e["event"] == "step_time"]
    if not times:
        raise ValueError("no steps recorded")
    if clean_step is None:
        # the mean, not the median: useful time is a sum of step times
        clean = [e["seconds"] for e in times if not e["writing"]]
        clean_step = statistics.fmean(clean)

    first = incarnations[0]
    first_step = next(e for e in first if e["event"] == "step_time")
    origin = first_step["t"] - first_step["seconds"]
    wall = end["t"] - origin
    startup = origin - start["t"]

    final_step = max(e["step"] for e in incarnations[-1] if e["event"] == "step_time")
    useful = final_step * clean_step
    lost = wall - useful

    stall = slowdown = redone = restart = 0.0
    for i, inc in enumerate(incarnations):
        if i + 1 < len(incarnations):
            kept_upto = next(e for e in incarnations[i + 1] if e["event"] == "restore")["step"]
        else:
            kept_upto = final_step
        steps = {e["step"]: e for e in inc if e["event"] == "step_time"}
        killed = i < len(kills)
        for e in inc:
            # the save taken at the kept step stalls after that step ended, so in a
            # killed life it falls inside the redone window and is counted there
            if e["event"] == "stall_end" and (e["step"] < kept_upto or (e["step"] == kept_upto and not killed)):
                stall += e["stall"]
            if e["event"] == "step_time" and e["step"] <= kept_upto and e["writing"]:
                slowdown += max(0.0, e["seconds"] - clean_step)
        if i < len(kills):
            kill_t = start["t"] + kills[i]["at"]
            anchor = steps.get(kept_upto)
            if anchor is not None:
                redone += kill_t - anchor["t"]
            elif steps:
                # nothing new was kept in this life: every step here is redone
                first_here = min(steps.values(), key=lambda e: e["step"])
                redone += kill_t - (first_here["t"] - first_here["seconds"])
            nxt = next((e for e in incarnations[i + 1] if e["event"] == "step_time"), None) \
                if i + 1 < len(incarnations) else None
            if nxt is not None:
                restart += (nxt["t"] - nxt["seconds"]) - kill_t

    hours = wall / 3600.0
    parts = {"stall": stall, "slowdown": slowdown, "redone": redone, "restart": restart}
    return {
        "wall_s": wall,
        "startup_s": startup,
        "kills": len(kills),
        "final_step": final_step,
        "clean_step_s": clean_step,
        "lost_s": lost,
        "lost_s_per_hour": lost / hours,
        **{f"{k}_s": v for k, v in parts.items()},
        "residual_s": lost - sum(parts.values()),
        "saves": sum(1 for e in events if e["event"] == "write_end"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="directory holding train.jsonl and supervisor.jsonl")
    p.add_argument("--clean-step", type=float, help="clean step time from a reference run on the same boot")
    args = p.parse_args(argv)
    d = Path(args.run_dir)
    result = account(d / "train.jsonl", d / "supervisor.jsonl", args.clean_step)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
