import pytest

from spanver import decode
from spanver.evaluate import metrics
from spanver.snap import snapped
from tools.sweep_hyst import flags_with
from tools.train_cv_feat import token_fpr
from tools.sweep_ragtruth30 import compute, distribution, prepare, within_fpr_limits


@pytest.mark.parametrize("start,extend,gate", [(0.25, 0.25, None), (0.5, 0.25, None), (0.7, 0.35, 0.45), (0.05, 0.05, None)])
def test_fast_metrics_match_reference(start, extend, gate):
    pairs = [
        {"pair_id": "one", "model_output": "Москва, Париж и Рим.", "spans": [{"start": 8, "end": 12, "type": "entity"}]},
        {"pair_id": "two", "model_output": "Без ошибок, верно.", "spans": []},
        {"pair_id": "three", "model_output": "Пример: один и два", "spans": [{"start": 0, "end": 5, "type": "scope"}]},
    ]
    cache = {
        "one": {"answer": pairs[0]["model_output"], "pieces": [(0, 6, 0.1), (8, 13, 0.8), (14, 15, 0.4), (16, 19, 0.6)]},
        "two": {"answer": pairs[1]["model_output"], "pieces": [(0, 3, 0.6), (4, 10, 0.1), (12, 17, 0.4)]},
        "three": {"answer": pairs[2]["model_output"], "pieces": [(0, 6, 0.9), (8, 12, 0.4), (13, 14, 0.3), (15, 18, 0.2)]},
    }

    def predict(pair):
        value = cache[pair["pair_id"]]
        if gate is not None and max(piece[2] for piece in value["pieces"]) < gate:
            return []
        flags = flags_with(value["pieces"], start, extend, 0.0)
        return snapped(decode.spans_from_flags(value["answer"], value["pieces"], flags, 2), value["answer"], "expand")

    reference = metrics(pairs, predict)
    actual = compute(prepare(pairs, cache), start, extend, gate)
    assert actual["token"] == {key: reference["token_level"][key] for key in ("precision", "recall", "f1")}
    assert actual["span"] == {key: reference["span_level"][key] for key in ("precision", "recall", "f1")}
    assert actual["pair"] == {key: reference["pair_level"][key] for key in ("precision", "recall", "f1")}
    assert actual["fpr_pair"] == reference["fpr"]
    assert actual["fpr_token"] == token_fpr(pairs, predict)


def test_fpr_limit_uses_exact_counts_not_rounded_values():
    row = {"support": {"flagged_negative_pairs": 2, "negative_pairs": 5, "flagged_negative_tokens": 40004, "negative_tokens": 100000}}
    assert not within_fpr_limits(row)
    row["support"]["flagged_negative_tokens"] = 40000
    assert within_fpr_limits(row)


def test_probability_histogram_labels_use_token_overlap():
    pair = {"pair_id": "x", "model_output": "Кот и пёс", "spans": [{"start": 1, "end": 1}]}
    cache = {"x": {"answer": pair["model_output"], "pieces": [(0, 3, 0.7), (4, 5, 0.3), (6, 9, 0.2)]}}
    histogram = distribution(prepare([pair], cache))
    assert histogram["positive"]["count"] == 1
    assert histogram["positive"]["quantiles"]["50%"] == pytest.approx(0.7)
    assert histogram["negative"]["count"] == 2
