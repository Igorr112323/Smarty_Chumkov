import unittest

from spanver.sentence_snap import snap_sentences


class Snap42Tests(unittest.TestCase):
    def test_b1_thirty_percent_is_inclusive(self):
        text = "abcdefghi."
        output = snap_sentences([{"start": 1, "end": 3}], text, "B1")
        self.assertEqual((output[0]["start"], output[0]["end"]), (0, 9))
        output = snap_sentences([{"start": 1, "end": 2}], text, "B1")
        self.assertEqual((output[0]["start"], output[0]["end"]), (1, 2))

    def test_b2_fifteen_percent_is_inclusive(self):
        text = "abcdefghijklmnopqrs."
        output = snap_sentences([{"start": 1, "end": 3}], text, "B2")
        self.assertEqual((output[0]["start"], output[0]["end"]), (0, 19))
        output = snap_sentences([{"start": 1, "end": 3}], text, "B1")
        self.assertEqual((output[0]["start"], output[0]["end"]), (1, 3))

    def test_b3_chains_and_uses_strict_gap(self):
        text = "Ааа. Ббб. Ввв." + " " * 20 + "Ггг."
        output = snap_sentences([{"start": 5, "end": 8}], text, "B3")
        self.assertEqual(output[0]["text"], "Ааа. Ббб. Ввв.")

    def test_b4_only_one_neighbor_each_side(self):
        text = "Ааа. Ббб. Ввв. Ггг. Ддд."
        output = snap_sentences([{"start": 10, "end": 13}], text, "B4")
        self.assertEqual(output[0]["text"], "Ббб. Ввв. Ггг.")

    def test_modes_do_not_change_an_unqualified_span(self):
        text = "abcdefghij. klmnopqrst."
        for mode in ("B1", "B2", "B3", "B4"):
            output = snap_sentences([{"start": 1, "end": 1}], text, mode)
            self.assertEqual((output[0]["start"], output[0]["end"]), (1, 1))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            snap_sentences([], "", "bad")


if __name__ == "__main__":
    unittest.main()
