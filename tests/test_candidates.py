import importlib.util
import json
from pathlib import Path

from spanver.candidates import build_candidates

ROOT = Path(__file__).resolve().parent.parent
TEXT = "Плотность составила 45% в 1990 году, возможно это так и не подтверждено."


def load_tool():
    spec = importlib.util.spec_from_file_location("make_candidates", ROOT / "tools" / "make_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(pair_id, docs, manual=False):
    return {
        "pair_id": pair_id,
        "domain": "sberquad",
        "query_id": "1",
        "query_text": "вопрос",
        "document_text": "документ",
        "doc_ids": docs,
        "model_output": TEXT,
        "spans": [],
        "label_source": "manual" if manual else "unlabeled",
    }


def test_pool_excludes_gold_documents():
    gold = [pair("sber000000", ["d1"], manual=True)]
    unlabeled = [pair("sbert000000", ["d1"]), pair("sbert000001", ["d2"]), pair("sbert000002", ["d3"])]
    selected, stats = build_candidates(gold, unlabeled, limit=10)
    ids = [item[0]["pair_id"] for item in selected]
    assert "sbert000000" not in ids
    assert stats["pool"] == 2
    assert stats["doc_overlap_with_gold"] == 0


def test_per_doc_cap():
    unlabeled = [pair(f"sbert{index:06d}", ["d9"]) for index in range(5)]
    selected, stats = build_candidates([], unlabeled, limit=10, cap=2)
    assert len(selected) == 2
    assert stats["max_per_doc"] == 2


def test_files_byte_identical(tmp_path):
    module = load_tool()
    gold = [pair("sber000000", ["d1"], manual=True)]
    unlabeled = [pair(f"sbert{index:06d}", [f"d{index}"]) for index in range(250)]
    (tmp_path / "gold.jsonl").write_text("".join(json.dumps(item) + "\n" for item in gold), encoding="utf-8")
    (tmp_path / "pairs_unlabeled.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in unlabeled), encoding="utf-8"
    )
    module.main(["--out", str(tmp_path)])
    first = (tmp_path / "candidates_200.jsonl").read_bytes()
    first_check = (tmp_path / "candidates_200_check.jsonl").read_bytes()
    module.main(["--out", str(tmp_path)])
    assert first == (tmp_path / "candidates_200.jsonl").read_bytes()
    assert first_check == (tmp_path / "candidates_200_check.jsonl").read_bytes()
