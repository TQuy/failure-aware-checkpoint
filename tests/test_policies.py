"""The running averages behind the adaptive policies."""

import unittest

from facp.policies import CheckFreq, Measured


def measured_after(stalls: list[float]) -> Measured:
    m = Measured()
    m.update("step_clean", 4.43)
    m.update("write", 59.0)
    for s in stalls:
        m.update("stall", s)
    return m


class FirstStall(unittest.TestCase):
    def test_first_stall_is_dropped(self):
        # the slow first save must not seed the average
        m = measured_after([4.9, 0.75])
        self.assertAlmostEqual(m.stall, 0.75)

    def test_checkfreq_settles_after_second_save(self):
        # measured on the T4: one slow first save, then 0.75 s
        policy = CheckFreq(0.01)
        self.assertEqual(policy.interval_steps(measured_after([4.9])), policy.first_steps)
        self.assertLessEqual(policy.interval_steps(measured_after([4.9, 0.75])), 18)

    def test_restored_state_counts(self):
        # after a restart, the averages loaded from the checkpoint are used
        m = Measured()
        m.load({"step_clean": 4.43, "write": 59.0, "stall": 0.75})
        self.assertTrue(m.ready())


if __name__ == "__main__":
    unittest.main()
