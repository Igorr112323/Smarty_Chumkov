import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from convert_rufacts import build_extra, convert_files, convert_record, doc_id

EVIDENCE = "Завод выпускал детали в 1990 году."
CLAIM = "Завод выпустил 1200 деталей в 1990 году."


def record(idx, claim=CLAIM, evidence=EVIDENCE, spans=None, label=1):
    facts = {}
    for position, (start, end, text) in enumerate(spans or []):
        facts[f"fact{position}"] = {"range": [start, end], "count": 1, "span": text}
    return {"idx": idx, "evidence": evidence, "claim": claim, "evidence_facts": {}, "claim_facts": facts, "label": label}


def test_localized_span_end_is_inclusive():
    start = CLAIM.find("1200")
    end = start + len("1200")
    kind, payload = convert_record(record(1, spans=[(start, end, "1200")]))
    assert kind == "localized"
    span = payload["spans"][0]
    assert span["start"] == start
    assert span["end"] == end - 1
    assert payload["model_output"][span["start"] : span["end"] + 1] == "1200"
    assert payload["document_text"] == EVIDENCE
    assert payload["doc_id"] == doc_id(EVIDENCE)
    assert len(payload["doc_id"]) == 16


def test_span_mismatch_is_rejected():
    start = CLAIM.find("1200")
    kind, payload = convert_record(record(2, spans=[(start, start + 4, "9999")]))
    assert kind is None
    assert payload == "span_mismatch"


def test_range_out_of_bounds_is_rejected():
    kind, payload = convert_record(record(3, spans=[(len(CLAIM), len(CLAIM) + 2, "го")]))
    assert kind is None
    assert payload == "range_out_of_bounds"


def test_pairlevel_and_negative_split():
    assert convert_record(record(4, label=1))[0] == "pairlevel"
    assert convert_record(record(5, label=0))[0] == "negative"


def test_convert_files_counts_and_reasons(tmp_path):
    source = tmp_path / "train_with_spans.json"
    start = CLAIM.find("1200")
    lines = [
        record(10, spans=[(start, start + 4, "1200")]),
        record(11, label=1),
        record(12, label=0),
        record(13, spans=[(start, start + 4, "9999")]),
    ]
    source.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")
    report = convert_files([source], tmp_path / "out", report_path=tmp_path / "report.json")
    assert report["records"] == 4
    assert report["localized"] == 1
    assert report["pairlevel"] == 1
    assert report["negative"] == 1
    assert report["rejected"] == 1
    assert report["reasons"] == {"span_mismatch": 1}
    assert report["spans"] == 1
    assert (tmp_path / "out" / "rufacts_localized.jsonl").is_file()


def test_extra_excludes_leaks():
    localized = [
        {"pair_id": "a", "doc_id": doc_id(EVIDENCE), "document_text": EVIDENCE, "model_output": CLAIM, "spans": []},
        {"pair_id": "b", "doc_id": doc_id("другой документ"), "document_text": "другой документ", "model_output": "ответ", "spans": []},
    ]
    ours = [{"pair_id": "mine", "doc_id": doc_id(EVIDENCE), "document_text": EVIDENCE, "model_output": CLAIM, "spans": []}]
    chosen, overlap = build_extra(localized, ours, 2)
    assert overlap == 1
    assert [pair["pair_id"] for pair in chosen] == ["b"]
    assert chosen[0]["split"] == "extra"
