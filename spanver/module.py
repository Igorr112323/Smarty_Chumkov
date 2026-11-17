import math
import re
import time

from .locate import STEPS, _apply
from .types import has_entity, has_number, has_scope, has_temporal

THRESHOLD = 0.98
IOU_THRESHOLD = 0.50
MIN_FRAGMENT = 10
MAX_SPANS = 3  # 52 из 699 пар имеют 2+ золотых спана; переопределяется параметром max_spans
NMS_IOU = 0.10
FEATURE_NAMES = ("number", "entity", "uncertainty", "negation", "doc_overlap")
MODEL_FEATURES = (
    "bias",
    "support",
    "stem_support",
    "ngram",
    "number_missing",
    "entity_missing",
    "numbers_absent",
    "entities_absent",
    "uncertainty",
    "negation",
    "refusal",
    "log_tokens",
    "relative_length",
    "whole_answer",
    "position",
    "doc_coverage",
)
WEIGHTS = (
    -2.723566,
    -4.670586,
    -4.085528,
    -6.658842,
    0.596939,
    1.848816,
    1.740967,
    5.600354,
    3.513775,
    -0.407672,
    2.232592,
    13.903439,
    21.343035,
    5.365126,
    -11.1283,
    -7.996501,
)

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)*")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")
CLAUSE_RE = re.compile(r"[^,;:]+[,;:]?")
DIGIT_TOKEN_RE = re.compile(r"\d")
CAPITAL_RE = re.compile(r"^[A-ZА-ЯЁ]")
SPACE_NUMBERS = re.compile(r"\d+[\s\u00a0]\d+|\d+")
REFUSAL_RE = re.compile(
    r"(нет (?:информации|данных|прямого ответа|сведений)|не (?:содержится|указано|упоминается|представлено|удалось найти)|"
    r"невозможно (?:определить|ответить|установить)|в (?:предоставленных|представленных|имеющихся) (?:документах|текстах) (?:нет|отсутствует)|"
    r"не хватает (?:информации|данных)|отсутствует (?:информация|в тексте))",
    re.I,
)

UNCERTAINTY_WORDS = (
    "вероятно",
    "возможно",
    "предположительно",
    "неизвестно",
    "не подтверждено",
    "по некоторым",
    "сообщается",
    "как утверждается",
    "по данным",
    "принято считать",
    "скорее всего",
)

NEGATION_WORDS = ("не", "нет", "никогда", "нигде", "ни", "нельзя", "отсутствует")

NUMERIC_WORDS = frozenset(
    "ноль один одна два две три четыре пять шесть семь восемь девять десять сто двести триста четыреста пятьсот тысяча "
    "тысячи тысяч миллион миллиона миллионов миллиард миллиарда миллиардов полтора половина треть четверть процент процента "
    "процентов".split()
)

STOPWORDS = frozenset(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее её мне было вот от меня еще ещё "
    "о из ему теперь когда даже ну вдруг ли если или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей "
    "может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того "
    "потому этого какой совсем ним здесь этом один почти мой тем чтобы нее неё сейчас были куда зачем всех никогда можно при "
    "наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою "
    "этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между является являются который которые которая "
    "которое которых котором этом этого этих также либо менее наиболее нею ими".split()
)


def normalize(text):
    normalized, _ = _apply(text or "", len(STEPS))
    return normalized


def tokens(text):
    return TOKEN_RE.findall(normalize(text or ""))


ENDINGS = (
    "иями", "ями", "ами", "ыми", "ими", "ов", "ев", "ий", "ый", "ая", "ое", "ые", "ей", "ом", "ем", "ах", "ях",
    "ую", "юю", "ия", "ии", "ой", "ть", "тся", "ться", "ешь", "ишь", "ует", "ают", "ять", "ать", "ить", "ла",
    "ло", "ли", "ет", "ут", "ют", "ит", "ат", "ят", "ся", "сь", "а", "е", "и", "о", "у", "ы", "ь", "я", "й",
)

EXCEPTIONS = {
    "россии": "росси",
    "россия": "росси",
    "люди": "люд",
    "человек": "человек",
    "годы": "год",
    "лет": "год",
    "года": "год",
    "году": "год",
    "время": "врем",
    "имени": "имен",
}

NUMBER_WORDS_DIGITS = {
    "ноль": "0", "один": "1", "одна": "1", "одно": "1", "два": "2", "две": "2", "три": "3",
    "четыре": "4", "пять": "5", "шесть": "6", "семь": "7", "восемь": "8", "девять": "9",
    "десять": "10", "одиннадцать": "11", "двенадцать": "12", "тринадцать": "13", "четырнадцать": "14",
    "пятнадцать": "15", "шестнадцать": "16", "семнадцать": "17", "восемнадцать": "18", "девятнадцать": "19",
    "двадцать": "20", "тридцать": "30", "сорок": "40", "пятьдесят": "50", "шестьдесят": "60",
    "семьдесят": "70", "восемьдесят": "80", "девяносто": "90", "сто": "100", "двести": "200",
    "триста": "300", "четыреста": "400", "пятьсот": "500", "тысяча": "1000", "тысячи": "1000",
    "тысяч": "1000", "миллион": "1000000", "миллиона": "1000000", "миллионов": "1000000",
    "миллиард": "1000000000", "миллиарда": "1000000000", "миллиардов": "1000000000",
}

ALIASES = {
    "км": "километр",
    "кг": "килограмм",
    "млн": "миллион",
    "млрд": "миллиард",
    "тыс": "тысяча",
    "гг": "год",
    "г": "год",
    "вв": "век",
    "в": "век",
    "проц": "процент",
    "руб": "рубль",
    "шт": "штука",
}


def stem(token):
    lowered = token.lower()
    if lowered in EXCEPTIONS:
        return EXCEPTIONS[lowered]
    if lowered in ALIASES:
        lowered = ALIASES[lowered]
    if lowered in NUMBER_WORDS_DIGITS:
        return NUMBER_WORDS_DIGITS[lowered]
    for ending in ENDINGS:
        if lowered.endswith(ending) and len(lowered) - len(ending) >= 4:
            return lowered[: -len(ending)]
    return lowered


def ngram_set(text, size=4):
    normalized = " " + normalize(text or "").lower() + " "
    return {normalized[index : index + size] for index in range(max(1, len(normalized) - size + 1))}


def document_context(document):
    tokens_found = document_token_set(document)
    stems = {stem(token) for token in tokens_found}
    return tokens_found, stems, ngram_set(document)


def document_token_set(document):
    found = set()
    for token in tokens(document):
        lowered = token.lower()
        found.add(lowered)
        found.add(stem(lowered))
    for token in SPACE_NUMBERS.findall(document or ""):
        found.add(token)
    return found


def marker_offsets(text, words):
    lowered = normalize(text).lower()
    found = []
    for word in words:
        start = lowered.find(word)
        while start != -1:
            found.append((start, start + len(word)))
            start = lowered.find(word, start + 1)
    return found


def token_features(text, document_tokens):
    normalized = normalize(text)
    raw = text or ""
    markers = marker_offsets(text, UNCERTAINTY_WORDS)
    negations = marker_offsets(text, NEGATION_WORDS)
    result = []
    for match in TOKEN_RE.finditer(normalized):
        token = match.group(0)
        lowered = token.lower()
        start, end = match.span()
        number = True if DIGIT_TOKEN_RE.search(token) else lowered in NUMERIC_WORDS
        original = raw[start:end] if end <= len(raw) else token
        entity = bool(CAPITAL_RE.match(original or "")) and start > 0
        uncertainty = any(left - 25 <= start <= right + 25 for left, right in markers)
        negation = any(left - 12 <= start <= right + 12 for left, right in negations)
        overlap = lowered in document_tokens or token in document_tokens or stem(lowered) in document_tokens
        result.append((token, number, entity, uncertainty, negation, overlap))
    return result


LONG_ANSWER = 20000


def candidate_spans(answer):
    text = answer or ""
    raw = []
    for match in SENTENCE_RE.finditer(text):
        raw.append((match.start(), match.end()))
    clauses = []
    for start, end in raw:
        piece = text[start:end]
        for match in CLAUSE_RE.finditer(piece):
            if match.group(0).strip():
                clauses.append((start + match.start(), start + match.end()))
    if len(text) > LONG_ANSWER:
        everything = raw + clauses
    else:
        merged = []
        for index in range(len(clauses) - 1):
            merged.append((clauses[index][0], clauses[index + 1][1]))
        everything = raw + clauses + merged + [(0, len(text))]
    seen = set()
    result = []
    for start, end in sorted(everything):
        piece = text[start:end]
        stripped = piece.strip()
        if len(stripped) < MIN_FRAGMENT:
            continue
        offset = piece.find(stripped)
        begin = start + (offset if offset >= 0 else 0)
        finish = begin + len(stripped)
        if (begin, finish) in seen:
            continue
        seen.add((begin, finish))
        result.append((begin, finish - 1, stripped))
    return result


def sigmoid(value):
    if value < -30:
        return 0.0
    if value > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def feature_vector(text, answer_length, document_tokens, whole=False, document_stems=None, document_ngrams=None, start=0, doc_length=0):
    features = token_features(text, document_tokens)
    stems = document_stems if document_stems is not None else {stem(token) for token in document_tokens}
    grams = document_ngrams if document_ngrams is not None else ngram_set(" ".join(document_tokens))
    total = 0.0
    matched = 0.0
    stemmed = 0.0
    numbers = 0
    numbers_missing = 0
    entities = 0
    entities_missing = 0
    uncertainty = 0.0
    negation = 0.0
    for token, number_flag, entity_flag, uncertainty_flag, negation_flag, overlap in features:
        total += 1.0
        if overlap:
            matched += 1.0
        if stem(token) in stems:
            stemmed += 1.0
        if number_flag:
            numbers += 1
            if not overlap:
                numbers_missing += 1
        if entity_flag:
            entities += 1
            if not overlap:
                entities_missing += 1
        if uncertainty_flag:
            uncertainty = 1.0
        if negation_flag:
            negation = 1.0
    support = matched / total if total else 0.0
    own_grams = ngram_set(text)
    coverage = len(own_grams & grams) / len(own_grams) if own_grams else 0.0
    values = [
        1.0,
        support,
        stemmed / total if total else 0.0,
        coverage,
        numbers_missing / numbers if numbers else 0.0,
        entities_missing / entities if entities else 0.0,
        math.log1p(numbers_missing),
        math.log1p(entities_missing),
        uncertainty,
        negation,
        1.0 if REFUSAL_RE.search(text) else 0.0,
        math.log1p(len(features)),
        len(text) / answer_length if answer_length else 0.0,
        1.0 if whole else 0.0,
        start / answer_length if answer_length else 0.0,
        math.log1p(doc_length),
    ]
    return values, support


def score(features, weights=None):
    vector = WEIGHTS if weights is None else weights
    total = 0.0
    for weight, value in zip(vector, features):
        total += weight * value
    return sigmoid(total)


def predict_type(text, support):
    if has_number(text):
        return "number"
    if has_temporal(text):
        return "temporal"
    if has_entity(text):
        return "entity"
    if has_scope(text):
        return "scope"
    if REFUSAL_RE.search(text):
        return "omission"
    if support < 0.35:
        return "unsupported_addition"
    return "contradiction"


CALIBRATION_POINTS = (
    (0.0, 0.0),
    (0.173, 0.25),
    (0.3321, 0.5),
    (0.4892, 0.75),
    (0.6542, 1.0),
)


def calibrate_share(raw):
    value = max(0.0, min(1.0, raw))
    points = CALIBRATION_POINTS
    if value <= points[0][0]:
        return round(points[0][1], 4)
    for index in range(1, len(points)):
        left_raw, left_share = points[index - 1]
        right_raw, right_share = points[index]
        if value <= right_raw:
            span = right_raw - left_raw
            if span <= 0:
                return round(right_share, 4)
            weight = (value - left_raw) / span
            return round(left_share + weight * (right_share - left_share), 4)
    return round(points[-1][1], 4)


def confidence(risk):
    span = max(1e-9, 1.0 - THRESHOLD)
    return round(min(1.0, max(0.0, 0.5 + (risk - THRESHOLD) / span * 0.5)), 4)


def sentence_spans(answer):
    text = answer or ""
    result = []
    for match in SENTENCE_RE.finditer(text):
        piece = match.group(0)
        stripped = piece.strip()
        if not stripped:
            continue
        offset = piece.find(stripped)
        start = match.start() + (offset if offset >= 0 else 0)
        result.append((start, start + len(stripped) - 1))
    return result


def select(scored, limit, nms_iou=NMS_IOU):
    ordered = sorted(scored, key=lambda item: (-item["risk"], item["start"], item["end"]))
    accepted = []
    for candidate in ordered:
        if len(accepted) >= limit:
            break
        overlaps = False
        for current in accepted:
            intersection = min(candidate["end"], current["end"]) - max(candidate["start"], current["start"]) + 1
            if intersection <= 0:
                continue
            union = max(candidate["end"], current["end"]) - min(candidate["start"], current["start"]) + 1
            if intersection / union > nms_iou:
                overlaps = True
                break
        if not overlaps:
            accepted.append(candidate)
    accepted.sort(key=lambda item: item["start"])
    return accepted


def verify(answer, document, threshold=None, weights=None, max_spans=None, nms_iou=None):
    started = time.perf_counter()
    text = answer or ""
    doc = document or ""
    limit = THRESHOLD if threshold is None else threshold
    document_tokens, document_stems, document_ngrams = document_context(doc)
    scored = []
    for start, end, fragment in candidate_spans(text):
        whole = start == 0 and end == len(text) - 1
        values, support = feature_vector(
            fragment, len(text), document_tokens, whole, document_stems, document_ngrams, start, len(doc)
        )
        risk = round(score(values, weights), 4)
        scored.append(
            {
                "start": start,
                "end": end,
                "risk": risk,
                "support": support,
                "text": fragment,
            }
        )
    chosen = [item for item in scored if item["risk"] >= limit]
    limit_spans = MAX_SPANS if max_spans is None else max_spans
    chosen = select(chosen, limit_spans, NMS_IOU if nms_iou is None else nms_iou)
    spans = []
    for item in chosen:
        spans.append(
            {
                "start": item["start"],
                "end": item["end"],
                "risk": item["risk"],
                "type": predict_type(item["text"], item["support"]),
                "confidence": confidence(item["risk"]),
            }
        )
    support_by_span = {(item["start"], item["end"]): item["support"] for item in scored}
    covered = 0.0
    total = 0.0
    for start, end in sentence_spans(text):
        length = end - start + 1
        total += length
        covered += length * (1.0 - support_by_span.get((start, end), 1.0))
    raw_share = covered / total if total else 0.0
    ai_share = calibrate_share(raw_share)
    elapsed = int(round((time.perf_counter() - started) * 1000))
    return {
        "spans": spans,
        "ai_share": ai_share,
        "meta": {
            "answer_len": len(text),
            "doc_len": len(doc),
            "n_spans": len(spans),
            "ai_share_raw": round(raw_share, 4),
            "elapsed_ms": elapsed,
        },
    }
