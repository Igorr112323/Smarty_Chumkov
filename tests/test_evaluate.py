from spanver.evaluate import evaluate_pairs, match_spans


def test_tp_ignores_type():
    predicted = [{"start": 0, "end": 4, "type": "number"}]
    gold = [{"start": 0, "end": 4, "type": "scope"}]
    matched, used, type_correct = match_spans(predicted, gold)
    assert matched == 1
    assert used == {0}
    assert type_correct == 0


def test_type_accuracy_counts_matches():
    predicted = [{"start": 0, "end": 4, "type": "scope"}, {"start": 10, "end": 14, "type": "number"}]
    gold = [{"start": 0, "end": 4, "type": "scope"}, {"start": 10, "end": 14, "type": "scope"}]
    matched, _, type_correct = match_spans(predicted, gold)
    assert matched == 2
    assert type_correct == 1


def test_iou_threshold_half():
    predicted = [{"start": 0, "end": 10, "type": "number"}]
    gold = [{"start": 0, "end": 4, "type": "number"}]
    matched, _, _ = match_spans(predicted, gold)
    assert matched == 0


def test_evaluate_pairs_shape():
    pairs = [
        {
            "model_output": "В 1990 году завод выпустил 1200 деталей.",
            "document_text": "Завод выпускает детали.",
            "spans": [{"start": 0, "end": 39, "type": "number"}],
        },
        {
            "model_output": "Завод выпускает детали.",
            "document_text": "Завод выпускает детали.",
            "spans": [],
        },
    ]
    metrics = evaluate_pairs(pairs)
    assert metrics["n_pairs"] == 2
    assert metrics["n_pairs_with_spans"] == 1
    assert metrics["n_neg_pairs"] == 1
    assert 0.0 <= metrics["fpr"] <= 1.0
    assert 0.0 <= metrics["type_accuracy"] <= 1.0
    assert metrics["iou_threshold"] == 0.5
