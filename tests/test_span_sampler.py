from spanver.span_detector import (
    BATCH_NEGATIVE,
    BATCH_POSITIVE,
    EPOCH_BATCHES,
    epoch_batches,
    token_stats,
)

POSITIVE = [f"pos{index:03d}" for index in range(193)]
NEGATIVE = [f"neg{index:03d}" for index in range(356)]


def test_batches_are_balanced():
    plan = epoch_batches(POSITIVE, NEGATIVE, seed=0)
    assert len(plan) == EPOCH_BATCHES
    for batch in plan:
        assert len(batch) == BATCH_POSITIVE + BATCH_NEGATIVE
        assert sum(1 for item in batch if item.startswith("pos")) == BATCH_POSITIVE
        assert sum(1 for item in batch if item.startswith("neg")) == BATCH_NEGATIVE


def test_every_positive_appears_in_epoch():
    plan = epoch_batches(POSITIVE, NEGATIVE, seed=0)
    seen = {item for batch in plan for item in batch if item.startswith("pos")}
    assert seen == set(POSITIVE)
    assert len([item for batch in plan for item in batch if item.startswith("pos")]) == EPOCH_BATCHES * BATCH_POSITIVE


def test_sampler_is_deterministic():
    first = epoch_batches(POSITIVE, NEGATIVE, seed=0)
    second = epoch_batches(POSITIVE, NEGATIVE, seed=0)
    assert first == second
    other = epoch_batches(POSITIVE, NEGATIVE, seed=1)
    assert first != other


def test_token_stats_matches_precision_recall_formula():
    cache = {
        "p1": {"answer": "abcdefgh", "pieces": [(0, 2, 0.9), (2, 4, 0.2), (4, 6, 0.8)]},
        "n1": {"answer": "abcdefgh", "pieces": [(0, 2, 0.1), (2, 4, 0.6)]},
    }
    pairs = [
        {"pair_id": "p1", "model_output": "abcdefgh", "spans": [{"start": 0, "end": 3}]},
        {"pair_id": "n1", "model_output": "abcdefgh", "spans": []},
    ]
    result = token_stats(cache, pairs, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["pair_fpr"] == 1.0
    expected = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    assert round(expected, 4) == result["f1"]
