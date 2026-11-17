import unittest

from tools.decision47 import decide


class Decision47Tests(unittest.TestCase):
    def test_success_requires_both_pooled_and_macro(self):
        self.assertEqual(decide(0.5, 0.501, 0.4, 90), "cv_success")
        self.assertEqual(decide(0.5, 0.499, 0.4, 90), "stop_metric_mismatch")
        self.assertEqual(decide(0.5, 0.501, 0.40001, 90), "stop_unspecified")

    def test_v5_requires_range_and_more_than_70_minutes(self):
        self.assertEqual(decide(0.48, 0.49, 0.41, 70.001), "run_v5")
        self.assertEqual(decide(0.4999, 0.49, 0.39, 70), "stop_budget")
        self.assertEqual(decide(0.4699, 0.46, 0.3, 100), "stop_keep_v2")
        self.assertEqual(decide(0.475, 0.49, 0.3, 100), "stop_unspecified")


if __name__ == "__main__":
    unittest.main()
