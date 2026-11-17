import unittest

from tools.fpr_bound43 import counts


class FprBoundTests(unittest.TestCase):
    def test_gold_negative_token_support(self):
        rows = [{"answer": "Один два три", "gold_spans": [{"start": 0, "end": 3}], "predicted_spans": [{"start": 0, "end": 6}]}, {"answer": "Четыре пять", "gold_spans": [], "predicted_spans": [{"start": 0, "end": 5}]}]
        result = counts(rows)
        self.assertEqual(result["negative_tokens"], 4)
        self.assertEqual(result["flagged_negative_tokens"], 2)
        self.assertEqual(result["fpr_token"], 0.5)
        self.assertEqual(result["fpr_pair"], 1.0)

    def test_absent_support_is_not_zero(self):
        result = counts([])
        self.assertIsNone(result["fpr_token"])
        self.assertIsNone(result["fpr_pair"])


if __name__ == "__main__":
    unittest.main()
