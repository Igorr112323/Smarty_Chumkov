from spanver.report import build_report


def gold_pair():
    return {
        "pair_id": "sber000000",
        "document_text": "документ",
        "model_output": "ответ",
        "spans": [
            {
                "text": "ответ",
                "start": 0,
                "end": 4,
                "type": "number",
                "type_src": "Unconfirmed",
                "evidence_ids": [],
            }
        ],
        "clean": False,
        "perturb": {"rel_position_mod": -1, "score_mod": 0.0},
    }


QC = {"coverage": 1.0, "out_of_bounds": 0, "span_text_mismatch": 0, "overlaps": 0, "nested": 0}
SPLIT = {
    "split_counts": {"test": 0, "val": 0, "train": 1, "unused": 0},
    "test_pairs_with_spans": 0,
    "components": 1,
    "max_component": 1,
    "doc_overlap_across_splits": 0,
}
COUNTERS = {"exact": 1, "normalized": 0, "unresolved": 0}


def test_report_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("SPANVER_BUILT_UTC", "2026-01-01T00:00:00Z")
    (tmp_path / "gold.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "pairs_unlabeled.jsonl").write_text("{}\n", encoding="utf-8")
    build_report(".", tmp_path, [gold_pair()], [], COUNTERS, QC, SPLIT)
    first = (tmp_path / "report.json").read_bytes()
    build_report(".", tmp_path, [gold_pair()], [], COUNTERS, QC, SPLIT)
    second = (tmp_path / "report.json").read_bytes()
    assert first == second


def test_report_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("SPANVER_BUILT_UTC", "2026-01-01T00:00:00Z")
    (tmp_path / "gold.jsonl").write_text("{}\n", encoding="utf-8")
    report, _ = build_report(".", tmp_path, [gold_pair()], [], COUNTERS, QC, SPLIT)
    assert report["pairs_total"] == 1
    assert report["spans_total"] == 1
    assert report["type_counts"]["number"] == 1
    assert report["type_src_counts"]["Unconfirmed"] == 1
    assert report["span_len_chars"]["max"] == 5
    assert report["files"]["pairs_unlabeled.jsonl"]["lines"] == 0
