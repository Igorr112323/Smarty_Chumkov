import pytest

from spanver import decode
from spanver.evaluate import metrics
from spanver.snap import snapped
from tools.stage_b31 import folds_for, tune
from tools.sweep_hyst import flags_with


def test_folds_require_699_pairs_and_keep_shared_documents_together():
    pairs = [{"pair_id": str(i), "doc_ids": ["shared"] if i < 3 else [str(i)]} for i in range(699)]
    folds = folds_for(pairs)
    assert sorted(len(fold) for fold in folds) == [139, 139, 140, 140, 141]
    assert sum(any(row["pair_id"] == "0" for row in fold) and any(row["pair_id"] == "2" for row in fold) for fold in folds) == 1
    with pytest.raises(ValueError):
        folds_for(pairs[:-1])


def test_tuning_uses_fold_and_returns_reference_type_accuracy():
    pairs = [{"pair_id": "1", "model_output": "Москва и Париж", "spans": [{"start": 0, "end": 5, "type": "entity"}]}, {"pair_id": "2", "model_output": "Простой ответ", "spans": []}]
    cache = {"1": {"answer": "Москва и Париж", "pieces": [(0, 6, 0.9), (7, 8, 0.2), (9, 14, 0.1)]}, "2": {"answer": "Простой ответ", "pieces": [(0, 7, 0.1), (8, 13, 0.1)]}}
    result = tune(pairs, cache)
    def predict(pair):
        value = cache[pair["pair_id"]]
        if max(score for _, _, score in value["pieces"]) < 0.45:
            return []
        flags = flags_with(value["pieces"], result["start"], result["extend"], 0.0)
        return snapped(decode.spans_from_flags(value["answer"], value["pieces"], flags, 2), value["answer"], "expand")
    assert result["measurements"]["type_accuracy"] == metrics(pairs, predict)["type_accuracy"]
    assert not result["eligible"]
    assert result["eligible_candidates"] == 0
