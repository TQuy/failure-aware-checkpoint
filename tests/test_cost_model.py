"""The cost model must reproduce the estimate table in the project proposal.

Inputs are the proposal's back-of-envelope assumptions for GPT-2 large on
one T4 (step 4.3 s, stall 1.9 s, write 55 s, 3% slowdown, restore 30 s),
with the save cost rounded to C = 3.5 s as in the proposal.
"""

import unittest

from facp.cost_model import (
    SaveCosts,
    checkfreq_steps,
    failure_aware_steps,
    lost_seconds_per_hour,
)

def proposal_costs() -> SaveCosts:
    # The proposal rounds C = 1.9 + 0.03 * 55 = 3.55 s down to 3.5 s, so the
    # slowdown here is set to give exactly that save cost.
    return SaveCosts(step=4.3, stall=1.9, write=55.0, slowdown=1.6 / 55.0, restore=30.0)


class ProposalTable(unittest.TestCase):
    def setUp(self):
        self.costs = proposal_costs()

    def test_save_cost(self):
        self.assertAlmostEqual(self.costs.save_cost, 3.5)

    def test_checkfreq_intervals(self):
        self.assertEqual(checkfreq_steps(self.costs, 0.01), 45)  # 193.5 s
        self.assertEqual(checkfreq_steps(self.costs, 0.05), 16)  # raised to the in-flight bound

    def test_failure_aware_intervals(self):
        self.assertEqual(failure_aware_steps(self.costs, 1 / 900), 18)  # 77 s
        self.assertEqual(failure_aware_steps(self.costs, 1 / 3600), 37)  # 159 s
        self.assertEqual(failure_aware_steps(self.costs, 1 / 21600), 90)  # 387 s

    def test_lost_seconds_match_table(self):
        table = {
            # failure every (s): (CheckFreq 1%, CheckFreq 5%, failure-aware)
            900: (792, 661, 658),
            3600: (247, 303, 244),
            21600: (95, 203, 79),
        }
        for mtbf, expected in table.items():
            rate = 1 / mtbf
            steps = (
                checkfreq_steps(self.costs, 0.01),
                checkfreq_steps(self.costs, 0.05),
                failure_aware_steps(self.costs, rate),
            )
            for k, want in zip(steps, expected):
                got = lost_seconds_per_hour(k * self.costs.step, self.costs, rate)
                self.assertAlmostEqual(got, want, delta=1.0, msg=f"M={mtbf}s, k={k}")

    def test_never_below_in_flight_bound(self):
        # a very high failure rate pushes Young's optimum under the bound
        k = failure_aware_steps(self.costs, 1 / 60)
        self.assertGreaterEqual(k * self.costs.step, self.costs.min_interval)


if __name__ == "__main__":
    unittest.main()
