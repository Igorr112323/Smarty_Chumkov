import re

NUMBER_WORDS = frozenset(
    {
        "ноль", "один", "одна", "одно", "одного", "одной", "два", "две", "двух", "три", "трех", "трёх",
        "четыре", "четырех", "четырёх", "пять", "пяти", "шесть", "шести", "семь", "семи", "восемь",
        "восьми", "девять", "девяти", "десять", "десяти", "одиннадцать", "двенадцать", "пятнадцать",
        "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят",
        "девяносто", "сто", "двести", "триста", "четыреста", "пятьсот", "тысяча", "тысячи", "тысяч",
        "миллион", "миллиона", "миллионов", "миллиард", "миллиарда", "миллиардов", "полтора", "полторы",
        "двое", "трое", "четверо", "оба", "обе", "половина", "треть", "четверть", "процент", "процента",
        "процентов",
    }
)

TEMPORAL_WORDS = frozenset(
    {
        "ранее", "позже", "период", "периода", "периоде", "год", "года", "году", "годом", "годы", "годах",
        "годов", "десятилетие", "десятилетия", "декада", "декады", "век", "века", "веке", "веков", "веках",
        "столетие", "столетия", "дата", "даты", "дату", "датой", "сегодня", "вчера", "завтра", "сначала",
        "потом", "современем", "ежегодно", "вековой",
    }
)

SCOPE_WORDS = frozenset(
    {
        "все", "всё", "всех", "всем", "всеми", "каждый", "каждая", "каждое", "каждые", "каждого",
        "каждой", "любой", "любая", "любое", "любые", "любого", "никогда", "всегда", "только",
        "исключительно", "значимо", "значимый", "значительная", "значительное", "существенно",
        "существенный", "существенная", "существенное", "полностью", "целиком", "единственный",
        "единственно",
    }
)

ORG_WORDS = frozenset(
    {
        "банк", "банка", "банке", "завод", "завода", "институт", "института", "компания", "компании",
        "министерство", "министерства", "академия", "академии", "федерация", "федерации", "университет",
        "университета", "корпорация", "корпорации", "фонд", "фонда", "агентство", "агентства", "союз",
        "союза", "общество", "общества", "партия", "партии", "музей", "музея", "театр", "театра",
    }
)

BASE_TYPES = {
    "Contradiction": "contradiction",
    "Missing": "omission",
    "Excess": "unsupported_addition",
    "Oversight": "unsupported_addition",
    "Unconfirmed": "unsupported_addition",
    "Partial": "unsupported_addition",
}

TYPE_ORDER = ("number", "temporal", "entity", "scope")
TYPE_KEYS = ("contradiction", "number", "entity", "omission", "unsupported_addition", "temporal", "scope")

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)*")
CAPITAL_RE = re.compile(r"^[A-ZА-ЯЁ]")
SENTENCE_END = ".!?;:…"


def tokens(text):
    return WORD_RE.findall(text or "")


def has_number(text):
    if DIGIT_RE.search(text or ""):
        return True
    for token in tokens(text):
        if token.lower() in NUMBER_WORDS:
            return True
    return False


def has_temporal(text):
    if YEAR_RE.search(text or "") or DATE_RE.search(text or ""):
        return True
    for token in tokens(text):
        if token.lower() in TEMPORAL_WORDS:
            return True
    return False


def has_entity(text):
    if not text:
        return False
    for token in tokens(text):
        if token.lower() in ORG_WORDS:
            return True
    for match in re.finditer(r"[A-ZА-ЯЁ][\w\-]*(?:[ \u00a0]+[A-ZА-ЯЁ][\w\-]*)+", text or ""):
        start = match.start()
        previous = text[:start].rstrip()
        at_start = not previous or previous[-1] in SENTENCE_END
        words = match.group(0).split()
        if len(words) >= 2 and all(CAPITAL_RE.match(word) for word in words) and not at_start:
            return True
        if len(words) >= 3 and all(CAPITAL_RE.match(word) for word in words) and at_start:
            return True
    return False


def has_scope(text):
    for token in tokens(text):
        if token.lower() in SCOPE_WORDS:
            return True
    return False


FEATURES = (
    ("number", has_number),
    ("temporal", has_temporal),
    ("entity", has_entity),
    ("scope", has_scope),
)


def features(text):
    found = []
    for name, probe in FEATURES:
        if probe(text):
            found.append(name)
    return found


def base_type(type_src):
    return BASE_TYPES.get(type_src, "unsupported_addition")


def classify(text, type_src):
    for name, probe in FEATURES:
        if probe(text):
            return name
    return base_type(type_src)
