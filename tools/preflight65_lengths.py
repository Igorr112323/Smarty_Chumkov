"""Гейт 1.2: токенизация всех кандидатов без обучения/инференса модели.

Только проверенные train/val 699 пар; тест 301 не читается. Сравниваются
действовавшая упаковка 63c и предлагаемая упаковка 65 на тех же кандидатах.
"""
import hashlib
import json
import re
import sys
import types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Только для загрузки pure-Python content_words/STOP из segment_model.
# Никакого вызова модели нет.
try:
    import torch  # noqa
except ModuleNotFoundError:
    torch = types.ModuleType('torch'); nn = types.ModuleType('torch.nn')
    nn.Module = object; torch.nn = nn
    sys.modules['torch'] = torch; sys.modules['torch.nn'] = nn
from spanver import segment_model
from tools.input65 import MAX_LENGTH, pack
from tools.segments63 import candidates_with_unions
from tokenizers import Tokenizer


def best_chunk(document, text, max_chars=1200):
    # Побуквенно повторяет train63.best_chunk; меняется только упаковка.
    if not document: return ''
    words = set(segment_model.content_words(text))
    if not words: return document[:max_chars]
    best = (0, 0)
    step = max(1, max_chars // 2)
    for start in range(0, max(1, len(document) - max_chars + 1), step):
        overlap = len(words & set(segment_model.content_words(document[start:start + max_chars])))
        if overlap > best[0]: best = (overlap, start)
    return document[best[1]:best[1] + max_chars]


def main():
    source = ROOT / 'artifacts/ours33.jsonl'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != '8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff':
        raise ValueError('неверный train/val набор')
    pairs = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
    tokenizer = Tokenizer.from_file(str(ROOT / 'models/rubert-tiny2/tokenizer.json'))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    counts = {'old_candidate_cut':0, 'new_candidate_cut':0, 'old_final_sep_lost':0,
              'new_document_cut':0, 'total':0, 'pairs':len(pairs)}
    per_pair_old = set(); per_pair_new = set(); worst = []
    counts['empty_questions'] = sum(not (p.get('query_text') or '').strip() for p in pairs)
    if counts['empty_questions']:
        raise ValueError('вопрос обязателен для всех 699 пар')
    for pair in pairs:
        answer, document = pair['model_output'] or '', pair['document_text'] or ''
        question = encode(pair.get('query_text') or '')[:80]
        for left, right in candidates_with_unions(answer):
            text = answer[left:right+1]
            tail = encode(text); doc = encode(best_chunk(document, text))[:140]
            visible = max(0, min(len(tail), 120, 256 - 3 - len(question) - len(doc)))
            old_cut = visible < len(tail)
            if old_cut:
                counts['old_candidate_cut'] += 1; per_pair_old.add(pair['pair_id'])
            if 4 + len(question) + len(doc) + min(len(tail),120) > 256:
                counts['old_final_sep_lost'] += 1
            try:
                packed = pack(question, doc, tail, 2, 3)
                assert len(packed) <= MAX_LENGTH and packed[-1] == 3
                assert packed[-len(tail)-1:-1] == tail if tail else packed[-1] == 3
                if len(packed) < 4 + len(question) + len(doc) + len(tail):
                    counts['new_document_cut'] += 1
            except ValueError:
                counts['new_candidate_cut'] += 1; per_pair_new.add(pair['pair_id'])
                worst.append({'pair_id':pair['pair_id'], 'start':left,
                              'end':right, 'candidate_tokens':len(tail), 'question_tokens':len(question)})
            counts['total'] += 1
    if counts['pairs'] != 699 or counts['total'] != 5973:
        raise ValueError('число пар/кандидатов расходится со стадией 1 промпта 63')
    counts.update({'gold_sha256':digest, 'tokenizer_sha256':hashlib.sha256((ROOT/'models/rubert-tiny2/tokenizer.json').read_bytes()).hexdigest(),
                   'old_candidate_cut_fraction':counts['old_candidate_cut']/counts['total'],
                   'new_candidate_cut_fraction':counts['new_candidate_cut']/counts['total'],
                   'old_pairs_with_cut':len(per_pair_old), 'new_pairs_with_cut':len(per_pair_new),
                   'new_context_tokens':MAX_LENGTH, 'overlength_candidates':worst,
                   'candidate_loss_check_pass':counts['new_candidate_cut']/counts['total'] <= .02,
                   'criterion_1_2_pass':False,
                   'gate_blocker':'не проверен полный обучающий шаг с оптимизатором и реальной упаковкой 512',
                   'model_inference':False, 'training':False, 'test301':False})
    target = ROOT / 'artifacts/length65_check.json'
    target.write_text(json.dumps(counts, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in counts.items() if k!='overlength_candidates'},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
