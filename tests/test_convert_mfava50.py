import unittest
from tools.convert_mfava50 import parse


class ConvertMFAVA50Test(unittest.TestCase):
    def test_exact_and_whitespace_substitution(self):
        spans, status = parse('abc <entity>foo\xa0bar</entity>!', 'abc foo bar!')
        self.assertEqual(status, 'ok_whitespace_substitution')
        self.assertEqual(spans, [{'type_src': 'entity', 'type': 'entity', 'start': 4, 'end': 10, 'text': 'foo bar'}])
        self.assertEqual(parse('a <invented>b</invented>', 'a b')[1], 'ok_exact')

    def test_reject_ambiguous(self):
        self.assertEqual(parse('a <entity>b</entity>', 'a  b')[1], 'unaligned_generated_text')
        self.assertEqual(parse('<entity>a <relation>b</relation></entity>', 'a b')[1], 'nested_annotations')
        self.assertEqual(parse('<conradictory>x</conradictory>', 'x')[1], 'unknown_tag')
        self.assertEqual(parse('<entity>x', 'x')[1], 'unclosed_tag')
        self.assertEqual(parse('hi </entity>', 'hi ')[1], 'unbalanced_tag')


if __name__ == '__main__':
    unittest.main()
