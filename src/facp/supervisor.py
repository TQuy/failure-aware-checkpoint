"""Failure injector and supervisor.

The supervisor starts the training command, kills it at seeded times, and
restarts it, which makes the job restore from its newest finished
checkpoint. Kill times are drawn once from an exponential distribution and
measured on the supervisor's wall clock, so every policy run with the same
seed and duration is killed at the same moments.

    python -m facp.supervisor --duration 7200 --mtbf 3600 --seed 1 \
        --record run/supervisor.jsonl -- python examples/train_gpt2.py ...
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import subprocess
import sys
import time

from facp.record import Record


def kill_times(duration: float, mtbf: float, seed: int) -> list[float]:
    """Seconds after the start at which the job is killed."""
    if mtbf <= 0:
        return []
    rng = random.Random(seed)
    times, t = [], 0.0
    while True:
        t += rng.expovariate(1.0 / mtbf)
        if t >= duration:
            return times
        times.append(t)


def run(cmd: list[str], duration: float, kills: list[float], record: Record) -> int:
    start = time.time()
    record.write("run_start", duration=duration, kills=kills, cmd=cmd)
    pending = list(kills)
    incarnation = 0
    while True:
        env = dict(os.environ, FACP_INCARNATION=str(incarnation))
        child = subprocess.Popen(cmd, env=env, start_new_session=True)
        record.write("child_start", incarnation=incarnation, pid=child.pid)
        while True:
            now = time.time() - start
            if child.poll() is not None:
                record.write("child_exit", incarnation=incarnation, code=child.returncode)
                record.write("run_end", reason="child_exit", elapsed=time.time() - start)
                return child.returncode
            if now >= duration:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                record.write("run_end", reason="duration", elapsed=time.time() - start)
                return 0
            if pending and now >= pending[0]:
                pending.pop(0)
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                record.write("kill", incarnation=incarnation, at=now)
                incarnation += 1
                break
            wait = min([duration - now] + [p - now for p in pending[:1]] + [0.5])
            time.sleep(max(0.01, wait))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, required=True, help="run length in seconds")
    p.add_argument("--mtbf", type=float, default=0.0, help="mean time between failures in seconds; 0 = no kills")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kill-at", type=float, nargs="*", help="explicit kill times in seconds (overrides --mtbf)")
    p.add_argument("--record", required=True)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        p.error("give the training command after --")
    kills = sorted(args.kill_at) if args.kill_at is not None else kill_times(args.duration, args.mtbf, args.seed)
    return run(cmd, args.duration, kills, Record(args.record, "supervisor"))


if __name__ == "__main__":
    sys.exit(main())
