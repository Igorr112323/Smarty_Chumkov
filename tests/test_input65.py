import unittest
from tools.input65 import pack


class Input65Tests(unittest.TestCase):
    def test_preserves_candidate_and_all_separators(self):
        q=list(range(10,90)); doc=list(range(100,350)); cand=list(range(400,700))
        ids=pack(q,doc,cand,2,3,512)
        self.assertLessEqual(len(ids),512)
        self.assertEqual(ids[-301:-1],cand)
        self.assertEqual(ids[-1],3)
        self.assertEqual(ids[:2],[2,10])
        self.assertEqual(ids[81],3)
        self.assertEqual(ids[-302],3)

    def test_too_long_fails_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, 'целиком не помещается'):
            pack(list(range(80)),[],list(range(429)),2,3,512)

    def test_preserves_shorter_document(self):
        self.assertEqual(pack([4,5],[6,7],[8,9],2,3,512),[2,4,5,3,6,7,3,8,9,3])


if __name__=='__main__': unittest.main()
