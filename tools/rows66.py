"""Общие строки кандидатов для теста памяти и финальной итерации 65/66.

Только упаковка изменена; генератор, признаки и метки линии 63 без изменений.
"""
from spanver import segment_model
from tools.input65 import pack
from tools.metrics59 import spans_of
from tools.segments63 import candidates_with_unions, label_candidates
from tools.train63 import TYPE_INDEX, best_chunk


def build_rows(tokenizer, pair, question, probabilities):
    answer = pair['model_output'] or ''
    document = pair.get('document_text') or ''
    marks = label_candidates(answer, candidates_with_unions(answer), spans_of(pair))
    question_words = segment_model.content_words(question)
    document_tokens = segment_model.document_token_set(document)
    head = tokenizer(question or '', add_special_tokens=False)['input_ids'][:80]
    total = len(marks)
    rows = []
    for rank, mark in enumerate(marks):
        text = answer[mark['start']:mark['end'] + 1]
        chunk = best_chunk(document, text)
        middle = tokenizer(chunk, add_special_tokens=False)['input_ids'][:140]
        tail = tokenizer(text, add_special_tokens=False)['input_ids']
        ids = pack(head, middle, tail, tokenizer.cls_token_id, tokenizer.sep_token_id)
        features = segment_model.candidate_features(
            answer, mark, question, document, question_words, document_tokens,
            [(tuple(piece[:2]), piece[2]) for piece in probabilities],
            {(mark['start'], mark['end']): rank / max(1, total)})
        rows.append({'input_ids': ids, 'attention_mask': [1]*len(ids), 'features': features,
                     'label': 1.0 if mark['positive'] else 0.0,
                     'type': TYPE_INDEX.get(mark['type'], 0), 'weight': len(text)/200.0,
                     'start': mark['start'], 'end': mark['end'], 'pair_id': pair['pair_id']})
    return rows
