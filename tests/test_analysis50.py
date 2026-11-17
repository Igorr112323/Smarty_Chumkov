import unittest
from tools.analysis50 import decision


class Decision50Test(unittest.TestCase):
    def test_accept_only_with_strict_f1_gain_and_both_fpr_limits(self):
        base = {'pooled': {'token_f1': 0.4998734625702283}}
        def item(f1, fpr, maximum):
            return {'pooled': {'token_f1': f1, 'fpr_token': fpr}, 'max_fpr_token': maximum}
        self.assertTrue(decision(item(0.5, 0.4, 0.555), base)['accepted'])
        self.assertFalse(decision(item(0.4998734625702283, 0.4, 0.555), base)['accepted'])
        self.assertTrue(decision(item(0.4998734625702283, 0.4, 0.555), base)['allow_v2'])
        self.assertFalse(decision(item(0.4998734625702282, 0.3, 0.5), base)['allow_v2'])
        self.assertFalse(decision(item(0.51, 0.4000000001, 0.5), base)['accepted'])
        self.assertFalse(decision(item(0.51, 0.3, 0.5550000001), base)['allow_v2'])


if __name__ == '__main__':
    unittest.main()
