import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import build_tiny

from spanver.stage_train import (
    build_table,
    collate_stage,
    make_stage_row,
    run_stage,
    stage_a_plans,
    stage_b_plans,
    stage_c_plans,
)

OURS = [f"ours{index:03d}" for index in range(40)]
HELPERS = [f"help{index:03d}" for index in range(120)]


def test_stage_a_plans_cover_ids():
    plans = stage_a_plans(OURS, 2, size=16)
    assert len(plans) == 2
    for plan in plans:
        assert sum(1 for batch in plan for item in batch) == len(OURS)
        for batch in plan:
            assert len(batch) <= 16
    assert len(plans[0]) == 3


def test_stage_b_plans_are_balanced():
    plans = stage_b_plans(OURS, HELPERS, 1)
    for batch in plans[0]:
        positive = sum(1 for item in batch if item.startswith("ours"))
        negative = sum(1 for item in batch if item.startswith("help"))
        assert positive == 8
        assert negative == 8


def test_stage_c_plans_are_mixed_and_deterministic():
    first = stage_c_plans(OURS, HELPERS, 1)
    second = stage_c_plans(OURS, HELPERS, 1)
    assert first == second
    for batch in first[0]:
        ours = sum(1 for item in batch if item.startswith("ours"))
        helpers = sum(1 for item in batch if item.startswith("help"))
        assert ours == 5
        assert helpers == 11
    covered = {item for batch in first[0] for item in batch if item.startswith("ours")}
    assert covered == set(OURS)


def test_stage_row_and_loss(tmp_path):
    directory = tmp_path / "tiny"
    build_tiny(directory, seed=0)
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(str(directory), num_labels=2, local_files_only=True)
    pair = {
        "pair_id": "p1",
        "model_output": "Завод выпустил 1200 деталей в 1990 году.",
        "document_text": "Завод выпускал детали в 1990 году.",
        "spans": [{"start": 15, "end": 18}],
        "label": 1,
    }
    row = make_stage_row(tokenizer, pair, "token")
    assert row is not None
    assert row["target"] == 1.0
    assert sum(row["answer_mask"]) > 0
    assert any(label == 1 for label in row["labels"])
    table = build_table(tokenizer, [pair], "token")
    assert "p1" in table
    plans = [[["p1", "p1"]]]
    history = run_stage(torch, model, tokenizer, table, plans, 1e-3, 1e-5)
    assert len(history) == 1
    assert history[0]["batches"] == 1
    assert history[0]["token_loss"] >= 0.0
    data = collate_stage(torch, [row, row], tokenizer.pad_token_id)
    assert data["input_ids"].shape[0] == 2
    assert data["targets"].tolist() == [1.0, 1.0]
