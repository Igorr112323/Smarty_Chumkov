import re

MIN_LENGTH = 10
MAX_FRAGMENT = 120
THRESHOLD = 0.35

NUMBER_UNIT_RE = re.compile(
    r"\d+[\s\u00a0]*(?:%|°|км|м|см|мм|кг|г|т|л|мл|руб|рублей|тыс|тысяч|млн|млрд|млн\.|шт|чел|человек|лет|года|год|дн|дней|час|часов|мин|минут|сек|секунд|раз|этаж|этажей)"
)
DIGITS_RE = re.compile(r"\d")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
TEMPORAL_WORDS = re.compile(r"\b(века|веке|веков|век|декада|декады|десятилетие|десятилетия|год|года|году|годы|годов)\b", re.I)
ORG_WORDS = re.compile(
    r"\b(банк|банка|завод|завода|институт|института|компания|компании|министерство|министерства|академия|академии|федерация|федерации|университет|университета|корпорация|корпорации|фонд|фонда|агентство|агентства)\b",
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
)
NEGATION_WORDS = ("не удалось", "не было", "нет", "не", "никогда")
CAPITAL_RE = re.compile(r"^[A-ZА-ЯЁ]")
FRAGMENT_RE = re.compile(r"[^.!?\n]+")
UNIT_WEIGHT = 0.35
TEMPORAL_WEIGHT = 0.30
ENTITY_WEIGHT = 0.30
UNCERTAINTY_WEIGHT = 0.25
UNCERTAINTY_CAP = 0.50
NEGATION_WEIGHT = 0.15


def fragments(text):
    for match in FRAGMENT_RE.finditer(text or ""):
        sentence = match.group(0)
        base = match.start()
        if len(sentence.strip()) > MAX_FRAGMENT:
            yield sentence, base
            continue
        offset = 0
        for part in sentence.split(","):
            yield part, base + offset
            offset += len(part) + 1


def has_entity(text):
    for match in re.finditer(r"[A-ZА-ЯЁ][\w\-]*(?:[ \u00a0]+[A-ZА-ЯЁ][\w\-]*)+", text):
        words = match.group(0).split()
        previous = text[: match.start()].rstrip()
        at_start = not previous or previous[-1] in ".!?;:…"
        if len(words) >= 2 and not at_start:
            return True
    return bool(ORG_WORDS.search(text))


def score_text(text):
    score = 0.0
    features = []
    if NUMBER_UNIT_RE.search(text) or DIGITS_RE.search(text):
        score += UNIT_WEIGHT
        features.append("number")
    if YEAR_RE.search(text) or DATE_RE.search(text) or TEMPORAL_WORDS.search(text):
        score += TEMPORAL_WEIGHT
        features.append("temporal")
    if has_entity(text):
        score += ENTITY_WEIGHT
        features.append("entity")
    lowered = text.lower()
    uncertainty = 0.0
    for word in UNCERTAINTY_WORDS:
        if word in lowered:
            uncertainty += UNCERTAINTY_WEIGHT
            features.append(f"uncertainty:{word}")
    if uncertainty:
        score += min(uncertainty, UNCERTAINTY_CAP)
    for word in NEGATION_WORDS:
        if re.search(rf"(?<![А-Яа-яЁёA-Za-z]){re.escape(word)}(?![А-Яа-яЁёA-Za-z])", lowered):
            score += NEGATION_WEIGHT
            features.append(f"negation:{word}")
            break
    return min(1.0, round(score, 4)), features


def detect(document_text, model_output):
    text = model_output or ""
    found = []
    seen = set()
    for raw, base in fragments(text):
        stripped = raw.strip()
        if len(stripped) < MIN_LENGTH:
            continue
        offset = raw.find(stripped)
        start = base + offset if offset >= 0 else base
        end = start + len(stripped) - 1
        if (start, end) in seen:
            continue
        score, features = score_text(stripped)
        if score < THRESHOLD:
            continue
        seen.add((start, end))
        found.append({"text": stripped, "start": start, "end": end, "score": score, "features": features})
    return found
