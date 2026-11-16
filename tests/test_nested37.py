import pytest

from tools.nested37 import compare_metrics, fine_choose, fine_grid
from tools.sweep_ragtruth30 import within_fpr_limits


def test_fine_grid_has_eleven_points_and_respects_boundaries():
    assert fine_grid(0.5) == [0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
    assert fine_grid(0.1) == [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    assert fine_grid(0.95)[-1] == 0.95


def test_fine_threshold_is_selected_on_inner_fpr_limits():
    pairs = [{"pair_id": "1", "model_output": "Москва и Париж", "spans": [{"start": 0, "end": 5, "type": "entity"}]}, {"pair_id": "2", "model_output": "Простой ответ", "spans": []}]
    cache = {"1": {"answer": pairs[0]["model_output"], "pieces": [(0, 6, 0.9), (7, 8, 0.2), (9, 14, 0.1)]}, "2": {"answer": pairs[1]["model_output"], "pieces": [(0, 7, 0.1), (8, 13, 0.1)]}}
    selected, rows = fine_choose(pairs, cache, 0.5)
    assert len(rows) == 11
    assert selected["start"] in fine_grid(0.5)
    assert selected["extend"] == selected["start"] / 2
    assert within_fpr_limits(selected)
    assert selected["token_f1_exact"] == max(row["token_f1_exact"] for row in rows if within_fpr_limits(row))


def test_missing_negative_support_is_not_an_fpr_guarantee():
    pairs = [{"pair_id": "1", "model_output": "Москва", "spans": [{"start": 0, "end": 5, "type": "entity"}]}]
    cache = {"1": {"answer": "Москва", "pieces": [(0, 6, 0.9)]}}
    with pytest.raises(ValueError):
        fine_choose(pairs, cache, 0.5)


def test_reproduction_reports_metric_differences_without_tolerance():
    assert compare_metrics({"token_f1": 0.5}, {"token_f1": 0.5}) == {}
    assert compare_metrics({"token_f1": 0.49}, {"token_f1": 0.5}) == {"token_f1": {"old": 0.5, "new": 0.49}}
