import unittest

from tools.diagnose46 import summarize, tally, without_snap


class Diagnose46Tests(unittest.TestCase):
    def test_hysteresis_and_gate_without_expansion(self):
        row = {"answer": "aa bb cc dd", "pieces": [[0, 2, 0.51], [3, 5, 0.31], [6, 8, 0.29], [9, 11, 0.51]], "gate_threshold": 0.45, "start_threshold": 0.5, "extend_threshold": 0.3}
        self.assertEqual([(s["start"], s["end"]) for s in without_snap(row)], [(0, 4), (9, 10)])
        row["gate_threshold"] = 0.6
        self.assertEqual(without_snap(row), [])

    def test_pooled_counts_not_macro_average(self):
        rows = [{"answer": "aa bb", "gold_spans": [], "predicted_spans": [{"start": 0, "end": 1}]}]
        counts = tally(rows)
        self.assertEqual(counts, {"negative_tokens": 2, "false_positive_tokens": 1, "negative_pairs": 1, "false_positive_pairs": 1})
        fields = ("token_precision", "token_recall", "token_f1", "exact_f1", "overlap_f1", "pair_f1", "fpr_pair", "fpr_token")
        folds = [{name: 0.0 for name in fields}, {name: 1.0 for name in fields}]
        output = summarize(folds, [counts, {"negative_tokens": 8, "false_positive_tokens": 0, "negative_pairs": 9, "false_positive_pairs": 0}])
        self.assertEqual(output["macro_means"]["fpr_token"], 0.5)
        self.assertEqual(output["pooled_fpr_token"], 0.1)
        self.assertEqual(output["pooled_fpr_pair"], 0.1)


if __name__ == "__main__":
    unittest.main()
