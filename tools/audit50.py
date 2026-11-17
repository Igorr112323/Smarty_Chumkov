import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.preflight_13 import our_rows

FILE = ROOT / 'Dataset/mFAVA_ru/test.json'
EXPECTED = '21464872b18b803959525063753fb8e92937b1dee0099a6e1bbdd7832c411b21'
TAG = re.compile(r'</?([a-z_]+)>')


def verified_rows(path=FILE):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED:
        raise ValueError(f'ошибка SHA mFAVA: {digest}')
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 1411:
        raise ValueError('ожидалось 1411 исходных записей')
    russian = [r for r in rows if r.get('language') == 'russian']
    if len(russian) != 267:
        raise ValueError('ожидалось 267 русских записей')
    return russian


def main():
    rows = verified_rows()
    tags = collections.Counter()
    consistency = collections.Counter()
    malformed = collections.Counter()
    types_in_generated = collections.Counter()
    positive = 0
    unique = collections.defaultdict(list)
    lengths = []
    for index, row in enumerate(rows):
        annotation, generated = row['gold_annotations'], row['generated_text']
        stack = []
        for match in TAG.finditer(annotation):
            kind = match.group(1)
            closing = match.group(0).startswith('</')
            tags[('close:' if closing else 'open:') + kind] += 1
            if closing:
                if not stack or stack[-1][0] != kind:
                    malformed['mismatched_close'] += 1
                else:
                    _, start = stack.pop()
                    lengths.append(len(TAG.sub('', annotation[start:match.start()])))
            else:
                if stack:
                    malformed['nested_open'] += 1
                stack.append((kind, match.end()))
        if stack:
            malformed['rows_with_unclosed'] += 1
        positive += bool(TAG.search(annotation))
        types_in_generated.update(m.group(0) for m in TAG.finditer(generated))
        clean = TAG.sub('', annotation)
        if clean == generated:
            consistency['exact'] += 1
        elif re.sub(r'\s+', ' ', clean).strip() == re.sub(r'\s+', ' ', generated).strip():
            consistency['after_collapsing_whitespace'] += 1
        elif re.sub(r'\s+', '', clean) == re.sub(r'\s+', '', generated):
            consistency['after_removing_whitespace'] += 1
        else:
            consistency['nonwhitespace_disagreement'] += 1
        unique[(row['references'], generated)].append(annotation)
    ours = our_rows()
    def norm(s):
        return re.sub(r'\s+', ' ', s).strip()
    mrefs = {norm(r['references']) for r in rows}
    manswers = {norm(r['generated_text']) for r in rows}
    our_refs = {norm(r['document_text']) for r in ours}
    our_answers = {norm(r['model_output']) for r in ours}
    lower_refs = [norm(r['references']).lower() for r in rows]
    segments = {text[i:i+120] for text in lower_refs for i in range(0, max(0, len(text)-119), 60)}
    snippet_overlap_pairs = sum(any(text[i:i+120] in segments for i in range(max(0, len(text)-119))) for text in (norm(r['document_text']).lower() for r in ours))
    result = {'file': str(FILE.relative_to(ROOT)), 'source_git_commit': 'e1ec868a681f7686c630dd68e06b7117a94f9d42', 'sha256': EXPECTED, 'sha256_verified': True, 'bytes': FILE.stat().st_size, 'total_rows': 1411, 'russian_rows': len(rows), 'unique_reference_answer_pairs': len(unique), 'duplicate_pair_groups': sum(len(v) > 1 for v in unique.values()), 'duplicate_groups_disagreeing_gold': sum(len(v) > 1 and len(set(v)) > 1 for v in unique.values()), 'tag_open_close_counts': dict(tags), 'rows_with_any_complete_tag': positive, 'matched_closing_tags': len(lengths), 'tagged_span_median_chars_before_alignment': statistics.median(lengths), 'malformed_nesting': dict(malformed), 'generated_text_complete_tags': dict(types_in_generated), 'tag_stripped_gold_vs_generated': dict(consistency), 'manual_alignment_not_performed': True, 'silver_used': False, 'ours_train_val_pairs': len(ours), 'normalized_exact_reference_overlap': len(mrefs & our_refs), 'normalized_exact_answer_overlap': len(manswers & our_answers), 'sampled_120_char_reference_snippet_overlap_pairs': snippet_overlap_pairs, 'overlap_limitation': 'полное нормализованное совпадение и 120-символьные окна mFAVA с шагом 60; иные частичные и семантические дубли не исключены', 'test301_evaluated': False, 'note': 'Строки с одинаковым входом имеют разные разметки; исходные теги содержат contradictory и опечатки. Метки не годятся для обучения без верифицированного сопоставления и согласования разметок.'}
    target = ROOT / 'artifacts/mfava50_audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
