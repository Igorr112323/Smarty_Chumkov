"""ПРОМПТ 59C: вероятности + векторы, чтобы ранжировать спаны по вопросу.

Аналог `lexfeat.batched_probabilities`, но возвращает ещё и скрытые состояния
энкодера: без них нельзя посчитать косинус вектора спана и вектора вопроса, а
нового обучения протокол C не допускает.

Вектор вопроса считается один раз на пару: вопрос отдельно кодируется
токенизатором, прогоняется через тот же энкодер, и берётся среднее по
субтокенам вопроса. Вектор спана — среднее по субтокенам прогона внутри ответа.
Средние без обучения, как зафиксировано в протоколе.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from spanver import lexfeat
from spanver.evaluate import token_positions
from spanver.span_detector import choose_chunks, encode_pair


def mean_vector(hidden, positions):
    """Среднее векторов по указанным позициям (список индексов)."""
    if not positions:
        return []
    picked = [hidden[index] for index in positions]
    size = len(picked[0])
    return [sum(vector[column] for vector in picked) / len(picked) for column in range(size)]


def question_vector(tokenizer, model, question, max_length=512):
    """Усреднённый вектор субтокенов вопроса."""
    if not (question or "").strip():
        return []
    encoded = tokenizer(question, add_special_tokens=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
    ids = [int(item) for item in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    keep = [index for index, (left, right) in enumerate(offsets) if right > left]
    if not keep:
        return []
    pad = tokenizer.pad_token_id
    features = [[[0.0] * lexfeat.FEATURE_COUNT for _ in ids]]
    with torch.no_grad():
        _, hidden = model(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=torch.tensor([[1] * len(ids)], dtype=torch.long),
            features=torch.tensor(features, dtype=torch.float),
            return_hidden=True,
        )
    return mean_vector(hidden[0].float(), keep)


def pair_probabilities_with_vectors(tokenizer, model, pair, top=4, max_length=512):
    """Вероятности, скрытые состояния и границы ответа для одной пары.

    Возвращает словарь с `answer`, `pieces` (как в поставке), `hidden`,
    `answer_index` (субтокен -> позиция в скрытых состояниях), `question_vector`
    и `chunk_spans` — границы ответа в каждом закодированном чанке.
    """
    answer = pair.get("model_output") or ""
    if not answer:
        return None
    chunks = choose_chunks(tokenizer, pair, top=top, max_length=max_length)
    encoded = []
    for chunk in chunks:
        item = encode_pair(tokenizer, chunk["text"], answer, max_length)
        if item is not None:
            encoded.append(item)
    if not encoded:
        return None
    rows = lexfeat.answer_features(answer, lexfeat.feature_document(pair))
    width = max(len(item["input_ids"]) for item in encoded)
    pad = tokenizer.pad_token_id
    batch_ids = [item["input_ids"] + [pad] * (width - len(item["input_ids"])) for item in encoded]
    batch_mask = [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in encoded]
    batch_features = []
    for item in encoded:
        values = [[0.0] * lexfeat.FEATURE_COUNT for _ in range(width)]
        for index, start, end in item["answer_tokens"]:
            values[index] = lexfeat.aligned(rows, start, end + 1)
        batch_features.append(values)
    with torch.no_grad():
        logits, hidden = model(
            input_ids=torch.tensor(batch_ids, dtype=torch.long),
            attention_mask=torch.tensor(batch_mask, dtype=torch.long),
            features=torch.tensor(batch_features, dtype=torch.float),
            return_hidden=True,
        )
        probabilities = torch.softmax(logits, dim=-1)[:, :, 1]
    best = {}
    best_hidden = {}
    for position, item in enumerate(encoded):
        for index, start, end in item["answer_tokens"]:
            value = float(probabilities[position][index])
            previous = best.get(start)
            if previous is None or value > previous[2]:
                best[start] = (start, end, value)
                best_hidden[start] = hidden[position][index].float().tolist()
    pieces = [best[key] for key in sorted(best)]
    return {
        "answer": answer,
        "pieces": pieces,
        "hidden_by_start": best_hidden,
        "question_vector": question_vector(tokenizer, model, lexfeat.question_of(pair), max_length),
    }


def run_vectors(answer, pieces, hidden_by_start, run):
    """Средний вектор субтокенов прогона."""
    first, last = run
    vectors = []
    for piece in pieces[first:last + 1]:
        value = hidden_by_start.get(piece[0])
        if value:
            vectors.append(value)
    if not vectors:
        return []
    size = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(size)]


def cascade_for_pair(scan, question, threshold, shelf=None):
    """Спан по каскаду C для одной пары (см. tools/cascade59.py)."""
    from tools.cascade59 import select_cascade

    pieces = scan["pieces"]
    if not pieces:
        return None
    runs = []
    # релевантность считается по каждому прогону, который может выиграть:
    # и по прошедшим порог, и по нижней полке — чтобы выбор был один и тот же
    # независимо от того, какая ступень включилась
    from tools.cascade59 import runs_of, SHELF_THRESHOLD

    candidates = set(runs_of(pieces, threshold))
    candidates |= set(runs_of(pieces, SHELF_THRESHOLD if shelf is None else shelf))
    vectors = {run: run_vectors(scan["answer"], pieces, scan["hidden_by_start"], run)
               for run in candidates}
    return select_cascade(scan["answer"], pieces, question, threshold,
                          vectors=(scan["question_vector"], vectors),
                          shelf=SHELF_THRESHOLD if shelf is None else shelf)
