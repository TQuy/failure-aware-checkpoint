"""End-to-end run on a tiny model: kill, restore, compare losses, account.

Needs PyTorch and transformers; skipped otherwise. Takes about a minute.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from facp.record import read
from facp.supervisor import kill_times

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class KillTimes(unittest.TestCase):
    def test_same_seed_same_times(self):
        self.assertEqual(kill_times(7200, 3600, 1), kill_times(7200, 3600, 1))
        self.assertNotEqual(kill_times(7200, 900, 1), kill_times(7200, 900, 2))

    def test_times_inside_run(self):
        times = kill_times(7200, 900, 3)
        self.assertTrue(all(0 < t < 7200 for t in times))
        self.assertEqual(times, sorted(times))


@unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
class EndToEnd(unittest.TestCase):
    def run_policy(self, policy: str, kills: list[float], duration: float = 40.0):
        tmp = Path(tempfile.mkdtemp(prefix="facp-"))
        train = [sys.executable, str(ROOT / "examples" / "train_gpt2.py"), "--model", "tiny",
                 "--policy", policy, "--ckpt-dir", str(tmp / "ckpt"), "--record", str(tmp / "train.jsonl"),
                 "--batch", "2", "--seq", "64", "--device", "cpu"]
        cmd = [sys.executable, "-m", "facp.supervisor", "--duration", str(duration),
               "--record", str(tmp / "supervisor.jsonl"), "--kill-at", *map(str, kills), "--", *train]
        subprocess.run(cmd, check=True, cwd=ROOT)
        out = subprocess.run([sys.executable, "-m", "facp.accounting", str(tmp)],
                             check=True, capture_output=True, text=True, cwd=ROOT)
        return tmp, json.loads(out.stdout)

    def check(self, tmp: Path, result: dict, kills: int):
        self.assertEqual(result["kills"], kills)
        self.assertGreater(result["saves"], 0)
        # every redone step reproduces the loss logged before the failure
        first: dict[int, float] = {}
        for e in read(tmp / "train.jsonl"):
            if e["event"] == "loss":
                if e["step"] in first:
                    self.assertAlmostEqual(e["loss"], first[e["step"]], places=4, msg=f"step {e['step']}")
                else:
                    first[e["step"]] = e["loss"]
        # the four parts explain the lost time to within 10% of the run
        self.assertLess(abs(result["residual_s"]), 0.1 * result["wall_s"])
        # old checkpoints are deleted, so a long run cannot fill the disk:
        # one finished checkpoint, plus the write the final stop cut off
        kept = [p for p in (tmp / "ckpt").iterdir() if p.name.startswith(("ckpt-", "dcp-"))]
        self.assertLessEqual(len(kept), 2, kept)

    def test_two_phase_with_kills(self):
        tmp, result = self.run_policy("fixed:20", [12.0, 26.0])
        self.check(tmp, result, 2)

    def test_dcp_with_kills(self):
        tmp, result = self.run_policy("dcp:20", [12.0, 26.0])
        self.check(tmp, result, 2)

    def test_adaptive_policies_run(self):
        for policy in ("checkfreq:0.05", "failure-aware:30"):
            tmp, result = self.run_policy(policy, [15.0], duration=30.0)
            self.check(tmp, result, 1)


if __name__ == "__main__":
    unittest.main()
