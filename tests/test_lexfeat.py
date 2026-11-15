from spanver import lexfeat


def test_answer_features_one_row_per_token():
    rows = lexfeat.answer_features("В 1990 году завод выпускал 42 детали.", "Завод выпускал детали")
    assert rows
    for _, _, flags in rows:
        assert len(flags) == lexfeat.FEATURE_COUNT
        assert all(flag in (0.0, 1.0) for flag in flags)


def test_number_and_overlap_flags():
    rows = lexfeat.answer_features("детали 1990 года", "детали выпускались")
    numbers = [flags[0] for _, _, flags in rows]
    overlaps = [flags[4] for _, _, flags in rows]
    assert max(numbers) == 1.0
    assert max(overlaps) == 1.0


def test_aligned_picks_maximum():
    rows = [(0, 5, [0.0, 1.0, 0.0, 0.0, 0.0]), (5, 10, [1.0, 0.0, 0.0, 0.0, 1.0])]
    values = lexfeat.aligned(rows, 2, 8)
    assert values == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert lexfeat.aligned(rows, 20, 30) == [0.0] * lexfeat.FEATURE_COUNT


def test_collate_shapes():
    rows = [
        {"pair_id": "a", "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [0, 1, -100], "answer_mask": [0, 1, 1], "features": [[0.0] * 5] * 3, "target": 1.0},
        {"pair_id": "b", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0, -100], "answer_mask": [0, 0], "features": [[1.0] * 5] * 2, "target": 0.0},
    ]
    batch = lexfeat.collate(rows, 0)
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["features"].shape) == (2, 3, lexfeat.FEATURE_COUNT)
    assert float(batch["targets"].sum()) == 1.0
    assert int(batch["attention_mask"][1].sum()) == 2
