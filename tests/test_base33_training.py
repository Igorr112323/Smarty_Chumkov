from tools.train_base33 import GRID, METRICS, grouped_folds, summarize_rows, tune


def test_grouped_folds_keep_shared_documents_together():
    pairs = [{"pair_id": str(index), "doc_ids": ["shared"] if index < 3 else [str(index)]} for index in range(30)]
    folds = grouped_folds(pairs, 3)
    assert sum(len(fold) for fold in folds) == 30
    assert sum({"0", "1", "2"}.issubset({row["pair_id"] for row in fold}) for fold in folds) == 1


def test_hysteresis_selection_uses_requested_grid_and_reference():
    assert GRID[0] == 0.1 and GRID[-1] == 0.95 and len(GRID) == 18
    pairs = [{"pair_id": "1", "model_output": "Москва и Париж", "spans": [{"start": 0, "end": 5, "type": "entity"}]}, {"pair_id": "2", "model_output": "Простой ответ", "spans": []}]
    cache = {"1": {"answer": "Москва и Париж", "pieces": [(0, 6, 0.9), (7, 8, 0.2), (9, 14, 0.1)]}, "2": {"answer": "Простой ответ", "pieces": [(0, 7, 0.1), (8, 13, 0.1)]}}
    result = tune(pairs, cache)
    assert result["start"] in GRID and result["extend"] in GRID
    assert result["extend"] <= result["start"]
    assert result["candidates"] == 171
    assert result["eligible"]
    assert 0 <= result["metrics"]["span_f1"] <= 1


def test_summary_uses_both_folds_and_population_std():
    values = (0.0, 0.2517)
    rows = [{"tuning": {"metrics": {name: value for name in METRICS}}} for value in values]
    result = summarize_rows(rows)
    assert result["token_f1"]["mean"] == round((values[0] + values[1]) / 2, 4)
    assert result["token_f1"]["std"] == result["token_f1"]["mean"]


def test_no_predictions_are_counted():
    pairs = [{"pair_id": "x", "model_output": "Нет ошибки.", "spans": []}]
    cache = {"x": {"answer": "Нет ошибки.", "pieces": [(0, 3, 0.01), (4, 10, 0.01)]}}
    result = tune(pairs, cache)
    assert result["predicted_pairs"] == 0
    assert result["predicted_spans"] == 0
