import unittest
from tools.metrics64 import measure


class Metrics64Tests(unittest.TestCase):
    def test_token_span_pair_and_counts(self):
        pairs = [{'pair_id': 'a', 'model_output': 'aa bb cc', 'spans': [{'start': 0, 'end': 3}]},
                 {'pair_id': 'b', 'model_output': 'dd ee', 'spans': []}]
        predictions = {'a': [{'start': 0, 'end': 1}], 'b': [{'start': 0, 'end': 1}]}
        score = measure(pairs, predictions)
        self.assertAlmostEqual(score['token']['f1'], .5)
        self.assertAlmostEqual(score['fpr_token'], 1/3)
        self.assertEqual(score['fpr_pair'], 1)
        self.assertEqual(score['span_f1_soft'], 2/3)
        self.assertEqual(score['span_f1_exact'], 0)
        self.assertEqual(score['pair_f1'], 2/3)
        self.assertEqual(score['predicted_pairs'], 2)


if __name__ == '__main__':
    unittest.main()
