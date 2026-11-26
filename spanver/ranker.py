"""Локальный ранкер кандидатов (чистый Python, без внешних зависимостей).

Веса обучены на train-сплите и 48 парах mFAVA_ru скриптом tools/train_ranker.py.
Вывод: гейт «есть ли в ответе неподтверждённые фрагменты» и ранжирование
кандидатов по риску. Используется бэкендом lexical по умолчанию.
"""

import json
import math
import re
from pathlib import Path

GATE = 0.50
THRESHOLD = 0.60
MAX_SPANS = 1
NMS_IOU = 0.10
FEATURE_COUNT = 37
PAIR_FEATURE_COUNT = 9
MODE_FEATURE_COUNT = 11
CUE_RE = re.compile(
    r"(считается|многие|некоторые|существует мнение|известно|по данным|по мнению|"
    r"по версии|утверждают|исследователи|по словам|согласно)",
    re.I,
)

NOINFO_RE = re.compile(
    r"(нет (информации|данных|сведений|прямого ответа)|"
    r"не (содержится|указано|упоминается|представлено|удалось найти)|"
    r"невозможно (определить|ответить|установить)|"
    r"отсутствует (информация|сведения)|"
    r"в (предоставленных|представленных) (документах|текстах))",
    re.I,
)

_WEIGHTS = None


def _model():
    global _WEIGHTS
    if _WEIGHTS is None:
        path = Path(__file__).resolve().with_name("ranker_weights.json")
        _WEIGHTS = json.loads(path.read_text(encoding="utf-8"))
    return _WEIGHTS


def sigmoid(value):
    value = max(-30.0, min(30.0, value))
    if value < -30.0:
        return 0.0
    if value > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _logit(features, params, expected):
    if len(features) != expected or len(params["w"]) != expected:
        raise ValueError(f"ranker: ожидается {expected} признаков")
    total = params["b"]
    for value, weight, mean, scale in zip(features, params["w"], params["mu"], params["sd"]):
        total += weight * ((value - mean) / scale if scale else 0.0)
    return sigmoid(total)


def _grams(text, size=5):
    normalized = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    return {normalized[i : i + size] for i in range(max(0, len(normalized) - size + 1))}


def features(text, start, end, answer, document, doc_tokens, doc_grams):
    """Признаки кандидата: 16 базовых + 21 дополнительный (без query_text)."""
    from .module import STOPWORDS, feature_vector, stem, tokens

    base, support = feature_vector(
        text, len(answer), doc_tokens, None, None, doc_grams, start, len(document)
    )
    toks = tokens(text)
    unique = {token.lower() for token in toks}
    missing = sum(1 for token in unique if token not in doc_tokens and stem(token) not in doc_tokens)
    five = _grams(text)
    covered_five = len(five & doc_grams) / len(five) if five else 0.0
    capitals = sum(1 for token in toks if token[:1].isupper())
    digits = sum(char.isdigit() for char in text)
    stop = sum(1 for token in unique if token in STOPWORDS) / len(unique) if unique else 0.0
    extra = [
        support,
        1.0 - support,
        len(unique) / max(1, len(toks)),
        math.log1p(len(toks)),
        math.log1p(len(text)),
        missing / max(1, len(unique)),
        capitals / max(1, len(toks)),
        digits / max(1, len(text)),
        stop,
        float(bool(NOINFO_RE.search(text))),
        float(text.strip().lower() in (document or "").lower()),
        covered_five,
        len(five & doc_grams),
        len(toks) / max(1, len(tokens(answer))),
        (len(answer) - end - 1) / max(1, len(answer)),
        base[3],
        base[5],
        base[7],
        base[11],
        base[12],
        base[14],
    ]
    return base + extra


def pair_features(rows, scores, answer):
    """Агрегаты по кандидатам для гейта пары.

    `rows` — либо словари с ключом `features` (рантайм), либо векторы признаков
    (обучение). Это одна и та же функция для train и inference.
    """
    vectors = [row["features"] if isinstance(row, dict) else row for row in rows]
    values = scores or [0.0]
    ordered = sorted(values)
    top3 = ordered[-min(3, len(ordered)) :]
    return [
        float(max(values)),
        float(sum(top3) / len(top3)),
        float(sum(value > 0.5 for value in values)),
        float(len(values)),
        float(sum(value > 0.2 for value in values)),
        float(sum(values) / len(values)),
        float(len(answer) > 500),
        max((row[28] for row in vectors), default=0.0),
        float(sum(1 for row in vectors if row[26] > 0.5)),
    ]


def iou(left_start, left_end, right_start, right_end):
    intersection = min(left_end, right_end) - max(left_start, right_start) + 1
    if intersection <= 0:
        return 0.0
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return intersection / union if union else 0.0


def mode_features(scored, answer, document):
    """Признаки выбора режима декодирования: global (RusHallu) или local (mFAVA)."""
    length = len(answer or "")
    local = [item for item in scored if not (item["start"] == 0 and item["end"] == length - 1)]
    whole = [item for item in scored if item["start"] == 0 and item["end"] == length - 1]
    probabilities = sorted((item["risk"] for item in local), reverse=True)
    cues = len(CUE_RE.findall(answer or ""))
    mean_length = sum(item["end"] - item["start"] + 1 for item in local) / len(local) if local else 0.0
    return [
        float(length),
        float(len(document or "")),
        float(len(scored)),
        float(len(local)),
        max(probabilities) if probabilities else 0.0,
        float(sum(value > 0.4 for value in probabilities)),
        float(sum(value > 0.5 for value in probabilities)),
        max((item["risk"] for item in whole), default=0.0),
        float(cues),
        cues / max(1.0, length / 1000.0),
        mean_length,
    ]


def select_spans(answer, document, gate=None, threshold=None, max_spans=None, nms_iou=None, collect_support=False):
    """Возвращает спаны контракта; по умолчанию выбирает режим по обученному гейту.

    Явные пороги отключают выбор режима и воспроизводят legacy-декодирование.
    """
    from .module import candidate_spans, document_context, predict_type, confidence
    from .local_ranker import logistic, predict_trees

    weights = _model()
    explicit = any(value is not None for value in (gate, threshold, max_spans, nms_iou))
    use_mode = not explicit and "mode" in weights and "local_hgb" in weights
    text = answer or ""
    if not text:
        return ([], {}) if collect_support else []
    doc = document or ""
    context = document_context(doc)
    doc_tokens, doc_grams = context[0], context[2]
    scored = []
    for start, end, fragment in candidate_spans(text):
        x = features(fragment, start, end, text, doc, doc_tokens, doc_grams)
        scored.append(
            {"start": start, "end": end, "text": fragment, "features": x}
        )
    local_mode = False
    if use_mode:
        for item in scored:
            item["risk"] = _logit(item["features"], weights["cand"], FEATURE_COUNT)
        mode_probability = logistic(mode_features(scored, text, doc), weights["mode"], MODE_FEATURE_COUNT)
        local_mode = mode_probability >= weights.get("mode_cut", 0.70)
    if local_mode:
        config = weights["local_config"]
        scored = []
        for start, end, fragment in candidate_spans(text):
            if start == 0 and end == len(text) - 1:
                continue
            x = features(fragment, start, end, text, doc, doc_tokens, doc_grams)
            scored.append(
                {
                    "start": start,
                    "end": end,
                    "text": fragment,
                    "features": x,
                    "risk": predict_trees(weights["local_hgb"], x),
                }
            )
        gate_probability = 1.0
        gate = -1.0
    else:
        for item in scored:
            item["risk"] = _logit(item["features"], weights["cand"], FEATURE_COUNT)
        config = weights.get("config", {})
        gate = config.get("gate", GATE) if gate is None else gate
        gate_probability = _logit(
            pair_features(scored, [item["risk"] for item in scored], text),
            weights["pair"],
            PAIR_FEATURE_COUNT,
        )
    support_map = {(item["start"], item["end"]): item["features"][16] for item in scored}
    if not scored:
        return ([], {}) if collect_support else []
    if gate_probability < gate:
        return ([], support_map) if collect_support else []
    threshold = config.get("threshold", THRESHOLD) if threshold is None else threshold
    max_spans = config.get("max_spans", MAX_SPANS) if max_spans is None else max_spans
    nms_iou = config.get("nms_iou", NMS_IOU) if nms_iou is None else nms_iou
    order = sorted(range(len(scored)), key=lambda index: (-scored[index]["risk"], scored[index]["start"]))
    accepted = []
    for index in order:
        if scored[index]["risk"] < threshold:
            break
        candidate = scored[index]
        if any(
            iou(candidate["start"], candidate["end"], other["start"], other["end"]) > nms_iou
            for other in accepted
        ):
            continue
        accepted.append(candidate)
        if len(accepted) >= max_spans:
            break
    result = []
    for item in accepted:
        result.append(
            {
                "start": item["start"],
                "end": item["end"],
                "risk": round(item["risk"], 4),
                "type": predict_type(item["text"], 0.0),
                "confidence": confidence(item["risk"]),
            }
        )
    if collect_support:
        return result, support_map
    return result
