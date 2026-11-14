import json


class AnswerError(ValueError):
    pass


def parse_answer(s, pair_id=None):
    if s is None:
        return []
    raw = s.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnswerError(f"pair {pair_id}: invalid answer JSON at position {exc.pos}") from exc
    if not isinstance(data, list):
        raise AnswerError(f"pair {pair_id}: answer is {type(data).__name__}, expected list")
    result = []
    for item in data:
        if not isinstance(item, dict):
            raise AnswerError(f"pair {pair_id}: answer item is {type(item).__name__}, expected object")
        text = item.get("span")
        kind = item.get("type")
        if text is None or kind is None:
            continue
        result.append({"text": text, "type": kind})
    return result
