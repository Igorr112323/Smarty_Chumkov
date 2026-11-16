import pytest

from tools.nested_cv35 import GRID, averaged, choose, evaluate, partition, verify_fast
from tools.summarize_nested36 import METRICS, summarize
from tools.train_base33 import grouped_folds


def test_nested_groups_are_disjoint_and_cover_all_pairs():
    pairs = [{"pair_id": str(index), "doc_ids": ["shared"] if index < 3 else [str(index)]} for index in range(50)]
    folds = grouped_folds(pairs, 5)
    for index in range(5):
        training, inner, outer, inner_index = partition(folds, index)
        assert inner_index == (index + 1) % 5
        sets = [{row["pair_id"] for row in group} for group in (training, inner, outer)]
        assert sum(map(len, sets)) == 50
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        assert sets[0] == {row["pair_id"] for i, fold in enumerate(folds) if i not in (index, inner_index) for row in fold}


def test_selection_is_inner_only_and_uses_fpr_and_half_threshold():
    pairs = [{"pair_id": "1", "model_output": "Москва и Париж", "spans": [{"start": 0, "end": 5, "type": "entity"}]}, {"pair_id": "2", "model_output": "Простой ответ", "spans": []}]
    cache = {"1": {"answer": pairs[0]["model_output"], "pieces": [(0, 6, 0.9), (7, 8, 0.2), (9, 14, 0.1)]}, "2": {"answer": pairs[1]["model_output"], "pieces": [(0, 7, 0.1), (8, 13, 0.1)]}}
    selected, rows = choose(pairs, cache)
    assert len(rows) == 18 and GRID[0] == 0.1 and GRID[-1] == 0.95
    assert all(row["extend"] == row["start"] / 2 for row in rows)
    assert selected["fpr_pair"] <= 0.4 and selected["fpr_token"] <= 0.4
    expected = evaluate(pairs, cache, {pair["pair_id"]: selected["start"] for pair in pairs})
    verify_fast(pairs, cache, selected["start"], expected)


def test_ensemble_is_three_distinct_probability_vectors():
    pair = {"pair_id": "p"}
    caches = [{"p": {"answer": "А Б", "pieces": [(0, 1, left), (2, 3, right)]}} for left, right in ((0.1, 0.3), (0.4, 0.6), (0.7, 0.9))]
    blended = averaged(pair, caches, (0, 1, 2))
    assert blended["pieces"][0][2] == pytest.approx(0.4)
    assert blended["pieces"][1][2] == pytest.approx(0.6)
    caches[1]["p"]["pieces"][0] = (0, 2, 0.4)
    with pytest.raises(ValueError):
        averaged(pair, caches, (0, 1, 2))


def test_summary_uses_population_standard_deviation():
    result = summarize([{metric: value for metric in METRICS} for value in (0.1, 0.3)])
    assert result["token_f1"] == {"mean": 0.2, "std": 0.1}
