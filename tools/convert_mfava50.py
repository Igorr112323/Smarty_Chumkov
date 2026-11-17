"""Conservative gold annotation audit; output is NOT an approved training set.

Whitespace-only text differences are mapped by character positions, only when
span boundaries are non-whitespace and every non-whitespace character matches.
Rows with nesting, unknown tags, malformed tags, or any other mismatch are
rejected; repeated inputs remain separate annotator rows, not a consensus.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.audit50 import verified_rows

TAGS = {'entity', 'relation', 'invented', 'contradictory', 'contradiction', 'subjective', 'unverifiable'}
PATTERN = re.compile(r'</?([a-z_]+)>')


def parse(annotation, generated):
    plain = []
    spans = []
    stack = []
    position = cursor = 0
    for match in PATTERN.finditer(annotation):
        piece = annotation[cursor:match.start()]
        plain.append(piece)
        position += len(piece)
        tag = match.group(1)
        close = match.group(0).startswith('</')
        if tag not in TAGS:
            return None, 'unknown_tag'
        if close:
            if not stack or stack[-1][0] != tag:
                return None, 'unbalanced_tag'
            kind, start = stack.pop()
            if position <= start:
                return None, 'empty_span'
            spans.append({'type_src': kind, 'type': kind, 'start': start, 'end': position - 1})
        else:
            if stack:
                return None, 'nested_annotations'
            stack.append((tag, position))
        cursor = match.end()
    if stack:
        return None, 'unclosed_tag'
    plain.append(annotation[cursor:])
    text = ''.join(plain)
    if '<' in text or '>' in text:
        return None, 'unparsed_angle_brackets'
    if text == generated:
        status = 'ok_exact'
    elif len(text) == len(generated) and all(a == b or (a.isspace() and b.isspace()) for a, b in zip(text, generated)):
        status = 'ok_whitespace_substitution'
    else:
        left = [i for i, char in enumerate(text) if not char.isspace()]
        right = [i for i, char in enumerate(generated) if not char.isspace()]
        if ''.join(text[i] for i in left) != ''.join(generated[i] for i in right):
            return None, 'unaligned_generated_text'
        positions = dict(zip(left, right))
        for span in spans:
            if span['start'] not in positions or span['end'] not in positions:
                return None, 'ambiguous_whitespace_boundary'
            span['start'], span['end'] = positions[span['start']], positions[span['end']]
        status = 'ok_whitespace_character_map'
    for span in spans:
        span['text'] = generated[span['start']:span['end']+1]
    return spans, status


def main():
    rows = verified_rows()
    converted, rejected = [], []
    for index, source in enumerate(rows):
        spans, reason = parse(source['gold_annotations'], source['generated_text'])
        if spans is None:
            rejected.append({'russian_row': index, 'reason': reason})
            continue
        reference = source['references']
        doc = hashlib.sha256(reference.encode('utf-8')).hexdigest()
        converted.append({'pair_id': f'mfava_ru_{index:03d}', 'doc_id': 'sha256:' + doc, 'doc_ids': ['sha256:' + doc], 'document_text': reference, 'model_output': source['generated_text'], 'spans': spans, 'gold_alignment': reason})
    result = {'approved_for_training': False, 'status': 'partial conservative conversion; no consensus or taxonomy mapping', 'source_sha256': '21464872b18b803959525063753fb8e92937b1dee0099a6e1bbdd7832c411b21', 'russian_rows': len(rows), 'converted_rows': len(converted), 'converted_spans': sum(len(r['spans']) for r in converted), 'conversion_status': dict(collections.Counter(r['gold_alignment'] for r in converted)), 'rejected_rows': len(rejected), 'rejection_reasons': dict(collections.Counter(r['reason'] for r in rejected)), 'rejected': rejected, 'warning': 'Повторные записи могут расходиться по разметке; типы mFAVA не приведены к типам проекта. Не использовать в обучении без согласования.', 'silver_used': False}
    path = ROOT / 'artifacts/mfava50_converted_partial.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in converted), encoding='utf-8')
    result['converted_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / 'artifacts/mfava50_conversion.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rejected'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
