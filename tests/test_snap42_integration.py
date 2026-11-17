import unittest

from tools.nested_cv35 import predict


class Snap42IntegrationTests(unittest.TestCase):
    def test_gate_is_unchanged_in_b3(self):
        value = {"answer": "Ааа. Ббб. Ввв.", "pieces": [[5, 8, 0.44]]}
        self.assertEqual(predict(value, 0.3, "B3"), [])

    def test_b3_is_opt_in_and_uses_baseline_decode(self):
        value = {"answer": "Ааа. Ббб. Ввв.", "pieces": [[5, 8, 0.8]]}
        original = predict(value, 0.5)
        transformed = predict(value, 0.5, "B3")
        self.assertEqual((original[0]["start"], original[0]["end"]), (4, 7))
        self.assertEqual((transformed[0]["start"], transformed[0]["end"]), (0, 13))


if __name__ == "__main__":
    unittest.main()
