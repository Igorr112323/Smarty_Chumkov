from spanver.splits import load_splits, ordered_pairs, split_ids


def test_canonical_counts():
    data = load_splits()
    assert len(data["test"]) == 301
    assert len(data["val"]) == 637
    assert data["counts"]["test"] == 301
    assert data["counts"]["val"] == 637


def test_gold_hash_recorded():
    data = load_splits()
    assert data["gold_sha256"] == "fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b"


def test_ids_unique_and_disjoint():
    data = load_splits()
    for name in ("test", "val", "train"):
        assert len(set(data[name])) == len(data[name])
    assert not set(data["test"]) & set(data["val"])
    assert not set(data["test"]) & set(data["train"])


def test_split_ids_matches_loader():
    assert split_ids("test") == load_splits()["test"]


def test_ordered_pairs_follows_ids():
    pairs = [{"pair_id": "b"}, {"pair_id": "a"}, {"pair_id": "c"}]
    ordered = ordered_pairs(pairs, ["a", "c", "z"])
    assert [pair["pair_id"] for pair in ordered] == ["a", "c"]
