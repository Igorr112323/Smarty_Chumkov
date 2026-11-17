import unittest

from tools.analysis48 import bootstrap, measures, pair_counts, quantile, totals
from tools.preflight_13 import components, our_keys
from tools.final48 import best_variant


class Analysis48Tests(unittest.TestCase):
    def test_best_uses_exact_pooled_and_tie_goes_to_v5(self):
        def item(value):
            return {"pooled_exact_from_counts": {"token_f1": value}}
        self.assertEqual(best_variant(item(0.4998734625702283), item(0.4998734625702283)), "v5")
        self.assertEqual(best_variant(item(0.4998734625702283), item(0.4998734625702282)), "v4")

    def test_counts_and_exact_metrics(self):
        row = {"answer": "aa bb cc dd", "gold_spans": [{"start": 0, "end": 4}], "predicted_spans": [{"start": 3, "end": 7}]}
        c = pair_counts(row)
        self.assertEqual(c[:4], [1, 1, 1, 2])
        self.assertEqual(measures(c)["token_f1"], 0.5)
        self.assertEqual(measures(c)["fpr_token"], 0.5)
        self.assertEqual(measures(c)["span_overlap_f1"], 1.0)
        self.assertEqual(measures(c)["span_exact_f1"], 0.0)

    def test_group_resampling_and_quantile(self):
        row = {"pair_id": "a", "doc_ids": ["x", "y"], "answer": "aa bb", "gold_spans": [{"start": 0, "end": 1}], "predicted_spans": [{"start": 0, "end": 1}]}
        group = components([row, {**row, "pair_id": "b", "doc_ids": ["y", "z"]}], our_keys)
        self.assertEqual(len(group), 1)
        self.assertEqual(totals(group[0])[:4], [2, 0, 0, 2])
        result = bootstrap(group, repeats=20)
        self.assertEqual(result["token_f1"]["ci_95"], [1.0, 1.0])
        self.assertEqual(result["fpr_token"]["width"], 0.0)
        self.assertEqual(quantile([1, 2, 3], 0.25), 1.5)


if __name__ == "__main__":
    unittest.main()
