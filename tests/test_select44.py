import unittest

from spanver.decode44 import predict44
from tools.select44 import DELTAS, GRID, admissible, base_curve, blend, compensation_curve


class Select44Tests(unittest.TestCase):
    def test_exact_fraction_038(self):
        self.assertTrue(admissible(50, 19, 100, 38))
        self.assertFalse(admissible(50, 20, 100, 38))
        self.assertFalse(admissible(50, 19, 100, 39))
        self.assertFalse(admissible(0, 0, 100, 0))

    def test_blending_same_offsets_and_gold(self):
        first = {"pair_id": "x", "answer": "тест", "gold_spans": [], "pieces": [[0, 4, 0.4]]}
        second = {"pair_id": "x", "answer": "тест", "gold_spans": [], "pieces": [[0, 4, 0.8]]}
        pairs, cache = blend([[first], [second]])
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(cache["x"]["pieces"][0][2], 0.6)
        second["pieces"][0][1] = 3
        with self.assertRaises(ValueError):
            blend([[first], [second]])

    def test_fixed_extend_and_gate(self):
        value = {"answer": "Ааа. Ббб.", "pieces": [[0, 3, 0.44], [5, 8, 0.4]]}
        self.assertEqual(predict44(value, 0.3, 0.15, True), [])
        value["pieces"][0][2] = 0.8
        self.assertEqual(len(predict44(value, 0.5, 0.25, True)), 1)
        self.assertEqual(DELTAS, (0.0, 0.02, 0.04, 0.06))
        self.assertEqual((GRID[0], GRID[-1], len(GRID)), (0.10, 0.95, 86))

    def test_base_selection_requires_each_inner_to_satisfy_both_caps(self):
        positive = {"pair_id": "p", "model_output": "Ошибка здесь.", "spans": [{"start": 0, "end": 5, "type": "contradiction"}]}
        negative = {"pair_id": "n", "model_output": "Нет ошибок.", "spans": []}
        cache = {"p": {"answer": positive["model_output"], "pieces": [[0, 6, 0.9], [7, 12, 0.05]]}, "n": {"answer": negative["model_output"], "pieces": [[0, 3, 0.1], [4, 9, 0.05]]}}
        selected, grid = base_curve([([positive, negative], cache)] * 4)
        self.assertEqual(len(grid), 86)
        self.assertIsNotNone(selected)
        self.assertTrue(selected["eligible"])
        self.assertLessEqual(selected["max_fpr_pair"], 0.38)
        self.assertLessEqual(selected["max_fpr_token"], 0.38)

    def test_compensation_retains_extend(self):
        positive = {"pair_id": "p", "model_output": "Ааа. Ббб.", "spans": [{"start": 0, "end": 3, "type": "contradiction"}]}
        negative = {"pair_id": "n", "model_output": "Нет ошибок.", "spans": []}
        cache = {"p": {"answer": positive["model_output"], "pieces": [[0, 3, 0.8], [5, 8, 0.1]]}, "n": {"answer": negative["model_output"], "pieces": [[0, 3, 0.1], [4, 9, 0.1]]}}
        selected, rows = compensation_curve([([positive, negative], cache)] * 4, 0.5, 0.25)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["extend"] == 0.25 for row in rows))
        self.assertIsNotNone(selected)


if __name__ == "__main__":
    unittest.main()
