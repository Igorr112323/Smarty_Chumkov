from spanver.evaluate import covered_tokens
from spanver.span_detector import answer_labels


def test_metric_marks_span_inside_word():
    text = "До (азан) после"
    assert covered_tokens(text, [(4, 7)]) == {1}


def test_metric_does_not_mark_adjacent_word():
    text = "a b"
    assert covered_tokens(text, [(2, 2)]) == {1}


def test_training_labels_use_exclusive_subword_end(monkeypatch):
    from spanver import lexfeat

    encoded = {"input_ids": [101, 10, 11, 12, 102], "answer_tokens": [(1, 0, 1), (2, 1, 5), (3, 5, 6)]}
    monkeypatch.setattr(lexfeat, "choose_chunks", lambda *args, **kwargs: [{"text": ""}])
    monkeypatch.setattr(lexfeat, "encode_pair", lambda *args, **kwargs: encoded)
    pair = {"pair_id": "boundary", "document_text": "", "model_output": "(азан)", "spans": [{"start": 1, "end": 4}]}
    row = lexfeat.build_row(None, pair)
    assert row["labels"] == [-100, 0, 1, 0, -100]
    assert row["labels"] == answer_labels(encoded, pair["spans"])


def test_inference_keeps_exclusive_token_end(monkeypatch):
    import torch
    from spanver import lexfeat, span_detector

    encoded = {"input_ids": [101, 10, 11, 12, 102], "answer_tokens": [(1, 0, 1), (2, 1, 5), (3, 5, 6)]}
    monkeypatch.setattr(lexfeat, "choose_chunks", lambda *args, **kwargs: [{"text": ""}])
    monkeypatch.setattr(lexfeat, "encode_pair", lambda *args, **kwargs: encoded)
    monkeypatch.setattr(span_detector, "choose_chunks", lambda *args, **kwargs: [{"text": ""}])
    monkeypatch.setattr(span_detector, "encode_pair", lambda *args, **kwargs: encoded)
    monkeypatch.setattr(lexfeat, "answer_features", lambda *args, **kwargs: [])

    class Tokenizer:
        pad_token_id = 0

    class Model:
        def __call__(self, input_ids, attention_mask, features):
            return torch.zeros((*input_ids.shape, 2))

    pair = {"pair_id": "p", "model_output": "(азан)", "document_text": ""}
    expected = [(0, 1), (1, 5), (5, 6)]
    single = lexfeat.pair_probabilities(Tokenizer(), Model(), pair)
    batch = lexfeat.batched_probabilities(Tokenizer(), Model(), [pair])
    assert [(left, right) for left, right, _ in single["pieces"]] == expected
    assert [(left, right) for left, right, _ in batch["p"]["pieces"]] == expected


def test_negative_has_no_positive_labels(monkeypatch):
    from spanver import lexfeat

    encoded = {"input_ids": [101, 10, 102], "answer_tokens": [(1, 0, 3)]}
    monkeypatch.setattr(lexfeat, "choose_chunks", lambda *args, **kwargs: [{"text": ""}])
    monkeypatch.setattr(lexfeat, "encode_pair", lambda *args, **kwargs: encoded)
    pair = {"pair_id": "negative", "document_text": "", "model_output": "кот", "spans": []}
    assert lexfeat.build_row(None, pair)["labels"] == [-100, 0, -100]
