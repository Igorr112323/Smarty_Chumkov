import unittest
from collections import Counter

from tools.diagnose_sentence41 import classify, summarize


class SentenceDiagnosisTests(unittest.TestCase):
    def test_exact_one_and_two_and_three_sentences(self):
        row = {"answer": "Один. Два! Три?", "gold_spans": [{"start": 0, "end": 4}, {"start": 0, "end": 9}, {"start": 0, "end": 14}, {"start": 1, "end": 4}]}
        self.assertEqual(classify(row), Counter({0: 1, 1: 1, 2: 1, 3: 1}))
        result = summarize(classify(row))
        self.assertEqual(result["single_percent"], 25.0)
        self.assertEqual(result["one_neighbor_percent"], 50.0)
        self.assertEqual(result["one_neighbor_each_side_percent"], 75.0)

    def test_each_neighbor_orientation_has_same_two_sentence_interval(self):
        row = {"answer": "А. Б.", "gold_spans": [{"start": 0, "end": 4}]}
        self.assertEqual(classify(row), Counter({2: 1}))

    def test_invalid_gold_span_is_rejected(self):
        row = {"answer": "А.", "gold_spans": [{"start": 0, "end": 2}]}
        with self.assertRaises(ValueError):
            classify(row)

    def test_empty_fold(self):
        self.assertIsNone(summarize(Counter())["single_percent"])


if __name__ == "__main__":
    unittest.main()
