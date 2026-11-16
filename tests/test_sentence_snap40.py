import unittest

from spanver.sentence_snap import sentence_intervals, snap_sentences


class SentenceSnapTests(unittest.TestCase):
    def test_intervals_trim_whitespace(self):
        self.assertEqual(sentence_intervals("  Первый.  Второй!\nТретий"), [(2, 8), (11, 17), (19, 24)])

    def test_thirty_percent_is_inclusive(self):
        text = "abcdefghi."
        self.assertEqual((snap_sentences([{"start": 2, "end": 4}], text)[0]["start"], snap_sentences([{"start": 2, "end": 4}], text)[0]["end"]), (0, 9))
        self.assertEqual((snap_sentences([{"start": 2, "end": 3}], text)[0]["start"], snap_sentences([{"start": 2, "end": 3}], text)[0]["end"]), (2, 3))

    def test_overlapping_expansions_are_merged(self):
        spans = [{"start": 0, "end": 3, "risk": 0.4}, {"start": 5, "end": 8, "risk": 0.8}]
        result = snap_sentences(spans, "abcdefghi.")
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["start"], result[0]["end"], result[0]["risk"]), (0, 9, 0.8))

    def test_spans_outside_answer_are_rejected(self):
        with self.assertRaises(ValueError):
            snap_sentences([{"start": -1, "end": 1}], "abcdef")


if __name__ == "__main__":
    unittest.main()
