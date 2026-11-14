import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import build_tiny

from spanver.evaluate import metrics
from spanver.model_features import HEAD_FEATURES, load_model, pair_features
from spanver.model_score import (
    THRESHOLD_GRID,
    ai_share,
    label_rows,
    make_predictor,
    scan_threshold,
    score_rows,
    train_head,
)

PAIRS = [
    {
        "pair_id": "tiny1",
        "model_output": "Завод выпустил 1200 деталей в 1990 году. Москва основана в 1147 году князем.",
        "document_text": "Завод выпускал детали в 1990 году. Москва — древний город, основанный князем.",
        "spans": [{"start": 0, "end": 41, "type": "number"}],
    },
    {
        "pair_id": "tiny2",
        "model_output": "Москва основана в 1147 году князем.",
        "document_text": "Москва — древний город, основанный князем.",
        "spans": [],
    },
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny_score")
    build_tiny(directory)
    return load_model(directory)


def test_pipeline_end_to_end(bundle):
    table = {}
    vectors = []
    labels = []
    for pair in PAIRS:
        rows = pair_features(bundle, pair["model_output"], pair["document_text"])
        table[pair["pair_id"]] = rows
        pair_labels = label_rows(rows, pair["spans"])
        vectors.extend(row["values"] for row in rows)
        labels.extend(pair_labels)
    assert vectors
    head = train_head(vectors, labels)
    assert len(head["weights"]) == len(HEAD_FEATURES)
    scored = {pair_id: score_rows(rows, head) for pair_id, rows in table.items()}
    for rows in scored.values():
        for row in rows:
            assert 0.0 <= row["risk"] <= 1.0
    result = metrics(PAIRS, make_predictor(scored, {pair["pair_id"]: pair["model_output"] for pair in PAIRS}, 0.5))
    assert result["n_pairs"] == 2
    assert 0.0 <= result["fpr"] <= 1.0
    assert 0.0 <= result["span_level"]["f1"] <= 1.0
    payload = json.dumps(result, ensure_ascii=False)
    assert json.loads(payload)["n_pairs"] == 2
    share = ai_share(scored["tiny1"], PAIRS[0]["model_output"])
    assert 0.0 <= share <= 1.0


def test_threshold_comes_from_grid(bundle):
    table = {}
    vectors = []
    labels = []
    for pair in PAIRS:
        rows = pair_features(bundle, pair["model_output"], pair["document_text"])
        table[pair["pair_id"]] = rows
        pair_labels = label_rows(rows, pair["spans"])
        vectors.extend(row["values"] for row in rows)
        labels.extend(pair_labels)
    head = train_head(vectors, labels)
    scored = {pair_id: score_rows(rows, head) for pair_id, rows in table.items()}
    best, history = scan_threshold(scored, PAIRS, THRESHOLD_GRID)
    assert best["threshold"] in THRESHOLD_GRID
    assert len(history) == len(THRESHOLD_GRID)
    assert 0.50 <= best["threshold"] <= 0.99


def test_predicted_spans_inside_answer(bundle):
    table = {}
    for pair in PAIRS:
        table[pair["pair_id"]] = pair_features(bundle, pair["model_output"], pair["document_text"])
    head = train_head([row["values"] for rows in table.values() for row in rows], [0] * sum(len(rows) for rows in table.values()))
    scored = {pair_id: score_rows(rows, head) for pair_id, rows in table.items()}
    predictor = make_predictor(scored, {pair["pair_id"]: pair["model_output"] for pair in PAIRS}, 0.0, max_spans=3)
    for pair in PAIRS:
        for span in predictor(pair):
            assert 0 <= span["start"] <= span["end"] < len(pair["model_output"])
            assert span["type"]
            assert 0.0 <= span["confidence"] <= 1.0
