import unittest

from tools.analyze_predictions40 import analyze


class PredictionAnalysisTests(unittest.TestCase):
    def test_exact_and_partial_overlap(self):
        rows = [{"gold_spans": [{"start": 0, "end": 4}, {"start": 7, "end": 10}], "predicted_spans": [{"start": 0, "end": 4}, {"start": 8, "end": 9}]}]
        result = analyze(rows)
        self.assertEqual(result["exact_tp"], 1)
        self.assertEqual(result["any_overlap_tp"], 2)
        self.assertEqual(result["overlapping_without_exact_gold"], 1)
        self.assertEqual(result["exact_f1"], 0.5)
        self.assertEqual(result["any_overlap_f1"], 1.0)
        self.assertEqual(result["pairs_with_multiple_predictions"], 1)

    def test_no_spans(self):
        result = analyze([{"gold_spans": [], "predicted_spans": []}])
        self.assertEqual(result["exact_f1"], 0.0)
        self.assertEqual(result["any_overlap_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
