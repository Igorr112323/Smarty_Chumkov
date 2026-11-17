import unittest

from tools.preflight_13 import components, id_digest, our_keys, rag_keys, split_groups


class PreflightTests(unittest.TestCase):
    def test_our_groups_connect_all_document_ids(self):
        rows = [
            {"pair_id": "a", "doc_ids": ["1", "2"]},
            {"pair_id": "b", "doc_ids": ["3", "2"]},
            {"pair_id": "c", "doc_ids": ["4"]},
        ]
        groups = components(rows, our_keys)
        self.assertEqual(sorted(len(group) for group in groups), [1, 2])

    def test_rag_groups_connect_same_text_and_doc_id(self):
        rows = [
            {"pair_id": "a", "doc_id": "1", "document_text": "same"},
            {"pair_id": "b", "doc_id": "2", "document_text": "same"},
            {"pair_id": "c", "doc_id": "2", "document_text": "other"},
            {"pair_id": "d", "doc_id": "3", "document_text": "third"},
        ]
        groups = components(rows, rag_keys)
        self.assertEqual(sorted(len(group) for group in groups), [1, 3])
        folds = split_groups(groups, 2)
        self.assertFalse({key for row in folds[0] for key in rag_keys(row)} & {key for row in folds[1] for key in rag_keys(row)})

    def test_splits_are_deterministic_and_digest_is_order_independent(self):
        groups = [[{"pair_id": str(index)}] for index in range(25)]
        first = split_groups(groups, 5)
        second = split_groups(groups, 5)
        self.assertEqual(first, second)
        self.assertEqual(id_digest(first[0]), id_digest(list(reversed(first[0]))))
        self.assertEqual(sum(map(len, first)), 25)


if __name__ == "__main__":
    unittest.main()
