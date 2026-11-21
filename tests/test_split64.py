import copy
import unittest
from tools.split64 import assign, gold_rows, make_record, validate


class DocumentSplit64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gold_rows()

    def test_complete_deterministic_no_overlap(self):
        result = make_record(self.rows)
        self.assertEqual(result, make_record(self.rows))
        self.assertEqual(result['pairs'], 1000)
        self.assertEqual(result['spans'], 423)
        self.assertEqual(result['components'], 943)
        self.assertEqual([f['doc_overlap'] for f in result['folds']], [0] * 10)
        self.assertTrue(all(98 <= f['pairs'] <= 102 for f in result['folds']))
        self.assertTrue(all(42 <= f['spans'] <= 43 for f in result['folds']))

    def test_overlap_is_a_hard_error(self):
        folds = copy.deepcopy(assign(self.rows))
        folds[0][0]['doc_ids'] = folds[1][0]['doc_ids']
        with self.assertRaisesRegex(ValueError, 'Пересечение doc_ids'):
            validate(self.rows, folds)


if __name__ == '__main__':
    unittest.main()
