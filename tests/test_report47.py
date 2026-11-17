import unittest

from tools.report47 import number, triple


class Report47Tests(unittest.TestCase):
    def test_metric_format(self):
        self.assertEqual(number(0.33104), "0,3310")
        self.assertEqual(triple({"precision": 0.1, "recall": 0.2, "f1": 0.3}), "0,1000 / 0,2000 / 0,3000")


if __name__ == "__main__":
    unittest.main()
