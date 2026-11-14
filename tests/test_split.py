import json

from spanver.split import TEST, TRAIN, UNUSED, VAL, assign, components, doc_overlap


def make_pair(pair_id, docs, manual=True, spans=None):
    return {
        "pair_id": pair_id,
        "doc_ids": docs,
        "label_source": "manual" if manual else "unlabeled",
        "spans": spans if spans is not None else ([{"text": "x", "start": 0, "end": 0}] if manual else []),
    }


def test_transitive_closure():
    pairs = [
        make_pair("a", ["d1"]),
        make_pair("b", ["d1", "d2"]),
        make_pair("c", ["d2"]),
        make_pair("d", ["d9"]),
    ]
    groups = components(pairs)
    merged = [group for group in groups if len(group) == 3]
    assert merged == [["a", "b", "c"]]


def test_no_doc_in_two_splits():
    pairs = [make_pair(f"p{index:03d}", [f"d{index // 2}"]) for index in range(60)]
    assignment, stats = assign(pairs)
    assert doc_overlap(pairs, assignment) == 0
    assert stats["split_counts"][TRAIN] + stats["split_counts"][VAL] + stats["split_counts"][TEST] + stats["split_counts"][UNUSED] == 60


def test_test_contains_only_manual():
    pairs = [make_pair(f"p{index:03d}", [f"d{index // 3}"], manual=(index % 2 == 0)) for index in range(120)]
    assignment, stats = assign(pairs)
    manual = {pair["pair_id"] for pair in pairs if pair["label_source"] == "manual"}
    for pair in pairs:
        if assignment[pair["pair_id"]] == TEST:
            assert pair["pair_id"] in manual
    assert stats["split_counts"][TEST] > 0


def test_unlabeled_in_test_component_goes_unused():
    pairs = [
        make_pair("g1", ["d1"]),
        make_pair("g2", ["d1"]),
        make_pair("g3", ["d1"]),
        make_pair("t1", ["d1"], manual=False),
    ]
    for _ in range(12):
        pairs.append(make_pair(f"m{len(pairs):03d}", [f"x{len(pairs)}"]))
    assignment, stats = assign(pairs)
    for pair in pairs:
        if assignment[pair["pair_id"]] == TEST:
            assert pair["label_source"] == "manual"


def test_deterministic_output():
    pairs = [make_pair(f"p{index:03d}", [f"d{index // 2}"]) for index in range(40)]
    first_assignment, first_stats = assign(pairs)
    second_assignment, second_stats = assign(pairs)
    assert json.dumps(first_assignment, sort_keys=True) == json.dumps(second_assignment, sort_keys=True)
    assert json.dumps(first_stats, sort_keys=True) == json.dumps(second_stats, sort_keys=True)
