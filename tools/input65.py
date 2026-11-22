"""Безопасная упаковка вопроса, документа и полного кандидата для промпта 65.

Не меняет нарезку кандидатов. Вместо обрезки всей последовательности по хвосту
сначала резервирует место кандидату и [SEP], потом сокращает только документ.
Если кандидат с вопросом не помещается, явно сообщает об ошибке.
"""
MAX_LENGTH = 512
MAX_QUESTION = 80
MAX_DOCUMENT = 140


def pack(question_ids, document_ids, candidate_ids, cls_id, sep_id, max_length=MAX_LENGTH):
    question = list(question_ids[:MAX_QUESTION])
    candidate = list(candidate_ids)
    capacity = max_length - len(question) - len(candidate) - 4
    if capacity < 0:
        raise ValueError('кандидат с вопросом целиком не помещается в контекст')
    document = list(document_ids[:min(MAX_DOCUMENT, capacity)])
    return [cls_id, *question, sep_id, *document, sep_id, *candidate, sep_id]
