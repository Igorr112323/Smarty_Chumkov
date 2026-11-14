import unicodedata


class LocateError(ValueError):
    pass


METHOD_EXACT = "exact"
METHOD_NORMALIZED = "normalized"
METHOD_UNRESOLVED = "unresolved"
METHOD_GIVEN = "given"

QUOTES = {"\u00ab": '"', "\u00bb": '"', "\u201e": '"', "\u201c": '"', "\u201d": '"', "\u201f": '"'}
DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2012": "-", "\u2015": "-"}
ZERO_WIDTH = frozenset("\u200b\u200c\u200d\ufeff\u2060\u00ad")


def _nfkc(text, index):
    chars = []
    mapping = []
    for char, origin in zip(text, index):
        for normalized in unicodedata.normalize("NFKC", char):
            chars.append(normalized)
            mapping.append(origin)
    return "".join(chars), mapping


def _spaces(text, index):
    chars = []
    mapping = []
    previous = False
    for char, origin in zip(text, index):
        if char.isspace():
            if not previous:
                chars.append(" ")
                mapping.append(origin)
            previous = True
        else:
            chars.append(char)
            mapping.append(origin)
            previous = False
    return "".join(chars), mapping


def _quotes(text, index):
    chars = []
    mapping = []
    for char, origin in zip(text, index):
        chars.append(QUOTES.get(char, char))
        mapping.append(origin)
    return "".join(chars), mapping


def _dashes(text, index):
    chars = []
    mapping = []
    for char, origin in zip(text, index):
        chars.append(DASHES.get(char, char))
        mapping.append(origin)
    return "".join(chars), mapping


def _zero_width(text, index):
    chars = []
    mapping = []
    for char, origin in zip(text, index):
        if char in ZERO_WIDTH or unicodedata.category(char) == "Cf":
            continue
        chars.append(char)
        mapping.append(origin)
    return "".join(chars), mapping


def _lower(text, index):
    chars = []
    mapping = []
    for char, origin in zip(text, index):
        for lowered in char.lower():
            chars.append(lowered)
            mapping.append(origin)
    return "".join(chars), mapping


STEPS = (
    ("nfkc", _nfkc),
    ("spaces", _spaces),
    ("quotes", _quotes),
    ("dashes", _dashes),
    ("zero_width", _zero_width),
    ("lower", _lower),
)


def _apply(text, count):
    index = list(range(len(text)))
    for _, step in STEPS[:count]:
        text, index = step(text, index)
    return text, index


def _taken(start, end, used):
    return (start, end) in used


def _validate(start, end, length):
    if not (0 <= start <= end < length):
        raise LocateError(f"span out of bounds: start={start} end={end} length={length}")


def _occurrences(haystack, needle, mapping, used):
    position = haystack.find(needle)
    while position != -1:
        if mapping is None:
            start = position
            end = position + len(needle) - 1
        else:
            start = mapping[position]
            end = mapping[position + len(needle) - 1]
        yield start, end
        position = haystack.find(needle, position + 1)


def locate(spans, model_output):
    text = model_output or ""
    used = []
    results = []
    cache = {}
    for span in spans:
        pattern = span.get("text") or ""
        given_start = span.get("start")
        given_end = span.get("end")
        if given_start is not None and given_end is not None:
            _validate(given_start, given_end, len(text))
            used.append((given_start, given_end))
            results.append({"text": pattern, "start": given_start, "end": given_end, "method": METHOD_GIVEN, "steps": 0})
            continue
        found = None
        method = METHOD_UNRESOLVED
        steps = 0
        if pattern:
            fallback = None
            for pair in _occurrences(text, pattern, None, None):
                if fallback is None:
                    fallback = pair
                if not _taken(pair[0], pair[1], used):
                    found = pair
                    method = METHOD_EXACT
                    break
            if found is None and fallback is not None:
                found = fallback
                method = METHOD_EXACT
            if found is None:
                for count in range(1, len(STEPS) + 1):
                    if count not in cache:
                        cache[count] = _apply(text, count)
                    haystack, mapping = cache[count]
                    needle, _ = _apply(pattern, count)
                    if not needle:
                        continue
                    inner = None
                    for pair in _occurrences(haystack, needle, mapping, None):
                        if inner is None:
                            inner = pair
                        if not _taken(pair[0], pair[1], used):
                            found = pair
                            method = METHOD_NORMALIZED
                            steps = count
                            break
                    if found is None and inner is not None:
                        found = inner
                        method = METHOD_NORMALIZED
                        steps = count
                    if found is not None:
                        break
        if found is None:
            results.append({"text": pattern, "start": None, "end": None, "method": METHOD_UNRESOLVED, "steps": 0})
            continue
        start, end = found
        _validate(start, end, len(text))
        used.append((start, end))
        results.append({"text": pattern, "start": start, "end": end, "method": method, "steps": steps})
    return results
