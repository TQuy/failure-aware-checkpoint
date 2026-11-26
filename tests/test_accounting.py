"""Accounting on hand-built records, where the right answer is known.

Every step takes exactly 1 s, so lost time is easy to work out by hand.
"""

import json
import tempfile
import unittest
from pathlib import Path

from facp.accounting import account


def write(path: Path, events: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def steps(first: int, last: int, t_first: float) -> list[dict]:
    """Steps first..last, 1 s each, the first one ending at t_first."""
    return [{"t": t_first + (k - first), "event": "step_time", "step": k, "seconds": 1.0, "writing": False}
            for k in range(first, last + 1)]


class AccountingEdgeCases(unittest.TestCase):
    def run_case(self, sup: list[dict], train: list[dict]) -> dict:
        d = Path(tempfile.mkdtemp())
        return account(write(d / "train.jsonl", train), write(d / "supervisor.jsonl", sup))

    def test_kill_before_first_step(self):
        # killed at 5 s while the model was still loading; no policy has acted yet
        sup = [{"t": 0, "event": "run_start"}, {"t": 5, "event": "kill", "at": 5},
               {"t": 100, "event": "run_end"}]
        train = [{"t": 0, "event": "start"},
                 {"t": 6, "event": "start"}, {"t": 8, "event": "restore", "step": 0}, *steps(1, 90, 11)]
        r = self.run_case(sup, train)
        self.assertEqual(r["early_kills"], 1)
        self.assertEqual(r["kills"], 0)
        self.assertAlmostEqual(r["wall_s"], 90)
        self.assertAlmostEqual(r["lost_s"], 0)
        self.assertAlmostEqual(r["residual_s"], 0)

    def test_kill_during_restore(self):
        # life 0 kept step 10 and was killed at 25; life 1 was killed at 30
        # before its restore finished; life 2 restored step 10
        sup = [{"t": 0, "event": "run_start"}, {"t": 25, "event": "kill", "at": 25},
               {"t": 30, "event": "kill", "at": 30}, {"t": 100, "event": "run_end"}]
        train = [{"t": 0, "event": "start"}, *steps(1, 20, 1),
                 {"t": 26, "event": "start"},
                 {"t": 31, "event": "start"}, {"t": 33, "event": "restore", "step": 10}, *steps(11, 76, 35)]
        r = self.run_case(sup, train)
        self.assertEqual(r["kills"], 2)
        self.assertAlmostEqual(r["lost_s"], 24)      # 100 s of wall time, 76 useful steps
        self.assertAlmostEqual(r["redone_s"], 15)    # step 10 ended at 10, killed at 25
        self.assertAlmostEqual(r["restart_s"], 9)    # 25 to 30, then 30 to 34
        self.assertAlmostEqual(r["residual_s"], 0)

    def test_kill_after_restore_before_a_step(self):
        # life 1 restored step 10 but was killed before its first step
        sup = [{"t": 0, "event": "run_start"}, {"t": 25, "event": "kill", "at": 25},
               {"t": 30, "event": "kill", "at": 30}, {"t": 100, "event": "run_end"}]
        train = [{"t": 0, "event": "start"}, *steps(1, 20, 1),
                 {"t": 26, "event": "start"}, {"t": 28, "event": "restore", "step": 10},
                 {"t": 31, "event": "start"}, {"t": 33, "event": "restore", "step": 10}, *steps(11, 76, 35)]
        r = self.run_case(sup, train)
        self.assertAlmostEqual(r["lost_s"], 24)
        self.assertAlmostEqual(r["redone_s"], 15)
        self.assertAlmostEqual(r["restart_s"], 9)
        self.assertAlmostEqual(r["residual_s"], 0)


if __name__ == "__main__":
    unittest.main()
