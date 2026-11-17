import unittest

from tools.report46 import number, row


class Report46Tests(unittest.TestCase):
    def test_decimals_and_column_order(self):
        self.assertEqual(number(0.12345), "0,1235")
        names = ("token_precision", "token_recall", "token_f1", "fpr_token", "fpr_pair", "exact_f1", "overlap_f1", "pair_f1")
        summary = {name: {"mean": index / 10, "max": index / 10 + 0.01} for index, name in enumerate(names)}
        values = [part.strip() for part in row("контроль", summary, "3 / 3 / 3", "20 с").strip("|").split("|")]
        self.assertEqual(values[:7], ["контроль", "0,0000", "0,1000", "0,2000", "0,3000", "0,3100", "0,4000"])
        self.assertEqual(len(values), 13)


if __name__ == "__main__":
    unittest.main()
