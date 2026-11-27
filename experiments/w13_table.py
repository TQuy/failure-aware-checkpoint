"""Build the Week 13 validation tables from the 27 November matrix.

    python experiments/w13_table.py results > results/w13/w13_table.md

Expects results/w13/seedN/m900/<cell>/ and results/w13/seedN/m3600/<cell>/
(folders ending in -aborted-* are runs cut short by a spot stop and are
skipped), and the Week 12 runs in results/pairN/results/seedN/ for the
CheckFreq first-stall before/after comparison.

The cost model is checked run by run. Its inputs (stall, write time,
slowdown, restore time) are measured once per checkpointer, pooled over all
40 runs. Its interval T is the run's measured mean time between saves.
Given the run's kills k and length W, the model predicts

    lost = W * C / T  +  k * (tau + T / 2)  +  k * R

(saving, redone work, restart). The expected form uses lambda * W in place
of k, which is the model's prediction before the kill times are drawn.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RATES = (900, 3600)
POLICIES = {  # label, cell folder (failure-aware takes the rate)
    "cf1": ("CheckFreq 1%", "checkfreq-0-01"),
    "cf5": ("CheckFreq 5%", "checkfreq-0-05"),
    "dcp": ("DCP every 64 steps", "dcp-64"),
    "fa": ("Failure-aware", "failure-aware-{m}"),
}
PAIRS = [("fa", "cf1"), ("fa", "cf5"), ("cf1", "cf5"), ("fa", "dcp")]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # two-sided t, by degrees of freedom
RATE_LABEL = {900: "one failure per 15 minutes", 3600: "one failure per 60 minutes"}


def events(d: Path) -> list[dict]:
    return [json.loads(line) for line in open(d / "train.jsonl")]


def run_facts(d: Path) -> dict:
    """Accounting plus what the model needs, measured from one run's record."""
    acc = json.loads((d / "accounting.json").read_text())
    ev = events(d)
    sup = [json.loads(line) for line in open(d / "supervisor.jsonl")]
    lives: list[list[dict]] = []
    for e in ev:
        if e["event"] == "start" or not lives:
            lives.append([])
        lives[-1].append(e)
    gaps, stalls, first_stalls = [], [], []
    for life in lives:
        s = [e for e in life if e["event"] == "stall_end"]
        gaps += [b["t"] - a["t"] for a, b in zip(s, s[1:])]
        if s:
            first_stalls.append(s[0]["stall"])
            stalls += [e["stall"] - e["waited"] for e in s[1:]]
    # for each kill: time from the first step of the killed life to the kill
    # (no new checkpoint can be older than that); 0 if the life had no step
    by_inc = {life[0].get("incarnation"): life for life in lives if life[0]["event"] == "start"}
    since_resume = []
    for k in (e for e in sup if e["event"] == "kill"):
        life = by_inc.get(k["incarnation"], [])
        first = next((e for e in life if e["event"] == "step_time"), None)
        since_resume.append(max(0.0, k["t"] - (first["t"] - first["seconds"])) if first else 0.0)
    since_resume = since_resume[acc.get("early_kills", 0):]
    steps = [e for e in ev if e["event"] == "step_time"]
    clean = [e["seconds"] for e in steps if not e["writing"]]
    writing = [e["seconds"] for e in steps if e["writing"]]
    return {
        **acc,
        "dir": d,
        "interval_s": statistics.fmean(gaps),
        "stalls": stalls,
        "first_stalls": first_stalls,
        "since_resume": since_resume,
        "waited_s": sum(e["waited"] for e in ev if e["event"] == "stall_end"),
        "writes": [e["write"] for e in ev if e["event"] == "write_end"],
        "slowdown_frac": statistics.fmean(writing) / statistics.fmean(clean) - 1 if writing else 0.0,
        "interval_steps": [e["steps"] for e in ev if e["event"] == "interval"],
        "kill_times": next(e for e in sup if e["event"] == "run_start")["kills"],
        "events": ev,
    }


def find_cell(root: Path, seed: int, rate: int, policy: str) -> Path | None:
    d = root / "w13" / f"seed{seed}" / f"m{rate}" / POLICIES[policy][1].format(m=rate)
    if (d / "accounting.json").exists() and "finished:" in (d / "cell.txt").read_text():
        return d
    return None


def ci(xs: list[float], digits: int = 0) -> str:
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return f"{m:+.{digits}f}"
    h = T95[len(xs) - 1] * statistics.stdev(xs) / len(xs) ** 0.5
    return f"{m:+.{digits}f} [{m - h:+.{digits}f}, {m + h:+.{digits}f}]"


def pooled_inputs(runs: list[dict]) -> dict:
    killed = [r for r in runs if r["kills"]]
    tau = statistics.fmean(w for r in runs for w in r["writes"])
    stall = statistics.median(s for r in runs for s in r["stalls"])
    slow = statistics.fmean(r["slowdown_frac"] for r in runs)
    return {
        "stall": stall,
        "write": tau,
        "slowdown": slow,
        "restore": sum(r["restart_s"] for r in killed) / sum(r["kills"] for r in killed),
        "C": stall + slow * tau,
    }


def predict(r: dict, p: dict, rate: int, expected: bool = False) -> dict:
    T, W = r["interval_s"], r["wall_s"]
    k = W / rate if expected else r["kills"]
    return {
        "save": W * p["C"] / T,
        "redone": k * (p["write"] + T / 2),
        "restart": k * p["restore"],
    }


def redone_restart_aware(r: dict, p: dict) -> float:
    """Redone work if each kill's checkpoint age is capped by the time since the last resume."""
    return sum(min(u, p["write"] + r["interval_s"] / 2) for u in r["since_resume"])


def main(root: Path) -> None:
    seeds = sorted(int(p.name[4:]) for p in (root / "w13").glob("seed*"))
    runs: dict[tuple[str, int, int], dict] = {}
    for s in seeds:
        for m in RATES:
            for pol in POLICIES:
                d = find_cell(root, s, m, pol)
                if d is not None:
                    runs[pol, m, s] = run_facts(d)
    kind = lambda pol: "dcp" if pol == "dcp" else "two-phase"  # noqa: E731
    inputs = {k: pooled_inputs([r for (pol, _, _), r in runs.items() if kind(pol) == k])
              for k in ("two-phase", "dcp")}

    print(f"Seeds: {', '.join(map(str, seeds))}. Runs found: {len(runs)} of {len(seeds) * 8}.\n")

    # 1. measured lost time
    for m in RATES:
        print(f"## Lost GPU-seconds per training hour, {RATE_LABEL[m]} (2-hour runs)\n")
        print("| Policy | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean |")
        print("|---|" + "---|" * (len(seeds) + 1))
        for pol, (label, _) in POLICIES.items():
            vals = [runs.get((pol, m, s)) for s in seeds]
            got = [v["lost_s_per_hour"] for v in vals if v]
            cells = " | ".join("–" if v is None else f"{v['lost_s_per_hour']:.0f}" for v in vals)
            print(f"| {label} | {cells} | {statistics.fmean(got):.0f} |")
        print("| Kills in the run | " + " | ".join(str(runs[("cf1", m, s)]["kills"]) for s in seeds) + " | |")
        print("| Interval chosen, steps (failure-aware) | " + " | ".join(
            f"{min(runs[('fa', m, s)]['interval_steps'][3:])}–{max(runs[('fa', m, s)]['interval_steps'][3:])}"
            for s in seeds) + " | |")
        print("| Interval chosen, steps (CheckFreq 1%) | " + " | ".join(
            f"{min(runs[('cf1', m, s)]['interval_steps'][3:])}–{max(runs[('cf1', m, s)]['interval_steps'][3:])}"
            for s in seeds) + " | |")
        print()

    # 2. paired differences
    print("## Paired differences over shared kill times (first minus second, seconds per hour, mean [95% interval])\n")
    for m in RATES:
        print(f"{RATE_LABEL[m].capitalize()}:\n")
        for a, b in PAIRS:
            d = [runs[a, m, s]["lost_s_per_hour"] - runs[b, m, s]["lost_s_per_hour"]
                 for s in seeds if (a, m, s) in runs and (b, m, s) in runs]
            base = statistics.fmean(runs[b, m, s]["lost_s_per_hour"] for s in seeds if (b, m, s) in runs)
            print(f"- {POLICIES[a][0]} minus {POLICIES[b][0]}: {ci(d)} (n = {len(d)}; "
                  f"{100 * statistics.fmean(d) / base:+.0f}% of {POLICIES[b][0]})")
        print()

    # 3. model inputs
    print("## Cost-model inputs, pooled over all runs of each checkpointer\n")
    print("| Input | Two-phase (CheckFreq, failure-aware) | DCP async_save |")
    print("|---|---|---|")
    for key, label, fmt in [("stall", "Stall per save, median (s)", "{:.2f}"),
                            ("write", "Background write, mean (s)", "{:.0f}"),
                            ("slowdown", "Step slowdown while writing", "{:.1%}"),
                            ("C", "Cost of one save C (s)", "{:.2f}"),
                            ("restore", "Restart per kill R (s)", "{:.0f}")]:
        print(f"| {label} | {fmt.format(inputs['two-phase'][key])} | {fmt.format(inputs['dcp'][key])} |")
    dw = [r["waited_s"] for (pol, _, _), r in runs.items() if pol == "dcp"]
    tw = [r["waited_s"] for (pol, _, _), r in runs.items() if pol != "dcp"]
    dT = [r["interval_s"] for (pol, _, _), r in runs.items() if pol == "dcp"]
    print(f"\nWaiting for the previous write, per 2-hour run: two-phase {min(tw):.0f}–{max(tw):.0f} s, "
          f"DCP {min(dw):.0f}–{max(dw):.0f} s. DCP's measured interval was {min(dT):.0f}–{max(dT):.0f} s.\n")

    # 4. model against measurement
    print("## Model against measurement (seconds, summed over seeds)\n")
    print("The model is given each run's measured interval and its kills.\n")
    print("The measured saving includes waiting for the previous write, which the model does not have.\n")
    print("| Rate | Policy | Kills | Saving: model / measured | Redone: model / measured | "
          "Redone, restart-aware model | Restart: model / measured | Total: model / measured | Total miss |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in RATES:
        for pol, (label, _) in POLICIES.items():
            rs = [runs[pol, m, s] for s in seeds if (pol, m, s) in runs]
            p = inputs[kind(pol)]
            pr = [predict(r, p, m) for r in rs]
            ps = {k: sum(x[k] for x in pr) for k in ("save", "redone", "restart")}
            ra = sum(redone_restart_aware(r, p) for r in rs)
            ms = {"save": sum(r["stall_s"] + r["slowdown_s"] for r in rs),
                  "redone": sum(r["redone_s"] for r in rs),
                  "restart": sum(r["restart_s"] for r in rs)}
            pt, mt = sum(ps.values()), sum(r["lost_s"] for r in rs)
            ram = f"{ra:.0f} ({100 * (ra / ms['redone'] - 1):+.0f}%)" if ms["redone"] else "0"
            red = f"{ps['redone']:.0f} / {ms['redone']:.0f} ({100 * (ps['redone'] / ms['redone'] - 1):+.0f}%)" \
                if ms["redone"] else "0 / 0"
            print(f"| {m // 60} min | {label} | {sum(r['kills'] for r in rs)} | {ps['save']:.0f} / {ms['save']:.0f} | "
                  f"{red} | {ram} | {ps['restart']:.0f} / {ms['restart']:.0f} | {pt:.0f} / {mt:.0f} | "
                  f"{100 * (pt / mt - 1):+.0f}% |")
    print()

    print("## Expected lost time before the kills are drawn (seconds per hour)\n")
    print("The model's 3600 L(T) at the set failure rate, against the measured mean over seeds "
          "(95% interval). The interval is wide because a 2-hour run sees only a few kills.\n")
    print("| Rate | Policy | Model | Measured |")
    print("|---|---|---|---|")
    for m in RATES:
        for pol, (label, _) in POLICIES.items():
            rs = [runs[pol, m, s] for s in seeds if (pol, m, s) in runs]
            pe = statistics.fmean(sum(predict(r, inputs[kind(pol)], m, expected=True).values()) / r["wall_s"] * 3600
                                  for r in rs)
            meas = ci([r["lost_s_per_hour"] for r in rs]).lstrip("+")
            print(f"| {m // 60} min | {label} | {pe:.0f} | {meas} |")
    killed = {m: [runs['cf1', m, s]['kills'] for s in seeds] for m in RATES}
    print(f"\nKills drawn: {killed[900]} at 15 minutes (8 expected per run), "
          f"{killed[3600]} at 60 minutes (2 expected).\n")

    # 5. first-stall before/after (Week 12 vs Week 13, seeds 1-3, one failure per hour)
    print("## CheckFreq 1% first-stall fix: before (Week 12) and after (Week 13), one failure per hour\n")
    print("| Seed | Same kill times | First stall (s) | Longest interval, before / after (steps) | "
          "Redone, before / after (s) | Lost per hour, before / after |")
    print("|---|---|---|---|---|---|")
    for s in (1, 2, 3):
        old_d = root / f"pair{s}" / "results" / f"seed{s}" / "checkfreq-0-01"
        if not (old_d / "accounting.json").exists() or ("cf1", 3600, s) not in runs:
            continue
        old, new = run_facts(old_d), runs["cf1", 3600, s]
        same = all(abs(a - b) < 1e-6 for a, b in zip(old["kill_times"], new["kill_times"])) \
            and len(old["kill_times"]) == len(new["kill_times"])
        print(f"| {s} | {'yes' if same else 'no'} ({len(new['kill_times'])} kills) | "
              f"{statistics.fmean(new['first_stalls']):.1f} | "
              f"{max(old['interval_steps'])} / {max(new['interval_steps'][1:])} | "
              f"{old['redone_s']:.0f} / {new['redone_s']:.0f} | "
              f"{old['lost_s_per_hour']:.0f} / {new['lost_s_per_hour']:.0f} |")
    print()

    # 6. checks
    worst = max(runs.values(), key=lambda r: abs(r["residual_s"]))
    redone = bad = 0
    for r in runs.values():
        first: dict[int, float] = {}
        for e in r["events"]:
            if e["event"] != "loss":
                continue
            if e["step"] in first:
                redone += 1
                bad += abs(e["loss"] - first[e["step"]]) > 1e-3
            else:
                first[e["step"]] = e["loss"]
    big = sorted(abs(r["residual_s"]) for r in runs.values())
    print("## Checks\n")
    print(f"- Accounting residual: under {big[-2]:.1f} s in all runs but one; the largest is "
          f"{worst['residual_s']:+.1f} s ({worst['dir'].relative_to(root)}).")
    print(f"- Kills before the first step: {sum(r['early_kills'] for r in runs.values())}.")
    print(f"- Redone steps that reproduced their logged loss: {redone - bad} of {redone}.")
    print(f"- Clean step time by seed: " + ", ".join(
        f"seed {s} {min(r['clean_step_s'] for (_, _, t), r in runs.items() if t == s):.2f}–"
        f"{max(r['clean_step_s'] for (_, _, t), r in runs.items() if t == s):.2f} s" for s in seeds) + ".")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
