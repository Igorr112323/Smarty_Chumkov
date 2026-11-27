"""Verification tests for the v19 synthetic Russian clause-level hallucination dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanver.qc import inspect as qc_inspect
from spanver.types import TYPE_KEYS
from tools.gen_ru_clause_v19 import (
    RusHalluBlacklist,
    norm_text,
    sha256_text,
    summarize_style_and_oracles,
    verify_zero_overlap,
)

DATASET_JSONL = ROOT / "artifacts" / "ru_clause_gen_v19.jsonl"
REPORT_JSON = ROOT / "artifacts" / "ru_clause_gen_v19.json"
GENERATOR_PY = ROOT / "tools" / "gen_ru_clause_v19.py"
DATASET_DIR = ROOT / "Dataset"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def report() -> dict:
    assert REPORT_JSON.exists(), f"Missing report: {REPORT_JSON}"
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    assert DATASET_JSONL.exists(), f"Missing dataset: {DATASET_JSONL}"
    rows = [json.loads(line) for line in DATASET_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "Dataset JSONL is empty"
    return rows


def test_sha256_and_report_consistency(report: dict, pairs: list[dict]) -> None:
    assert report["version"] == "ru_clause_gen_v19"
    assert report["programmatic_only"] is True
    assert report["manual_annotation"] is False
    assert report["flip_augmentation_used"] is False
    assert report["gates"]["all_passed"] is True

    assert _file_sha256(DATASET_JSONL) == report["dataset_sha256"]
    assert _file_sha256(GENERATOR_PY) == report["generator_script_sha256"]
    assert len(pairs) == report["stats"]["n_pairs"]
    assert sum(len(p["spans"]) for p in pairs) == report["stats"]["n_spans"]


def test_span_schema_coordinates_and_qc(pairs: list[dict]) -> None:
    qc = qc_inspect(pairs)
    assert qc["ok"] is True
    assert qc["control_pairs"] == 0
    assert qc["out_of_bounds"] == 0
    assert qc["span_text_mismatch"] == 0
    assert qc["overlaps"] == 0
    assert qc["nested"] == 0
    assert qc["coverage"] == 1.0

    required_span_keys = {"text", "start", "end", "type", "style", "position", "risk", "confidence", "donor_corpus"}
    valid_types = set(TYPE_KEYS)
    valid_styles = {"whole", "clause", "short"}

    for pair in pairs:
        ans = pair["model_output"]
        assert isinstance(pair["document_text"], str) and len(pair["document_text"]) >= 50
        assert isinstance(ans, str) and len(ans) >= 20
        assert pair["clean"] == (len(pair["spans"]) == 0)

        for sp in pair["spans"]:
            assert required_span_keys.issubset(sp.keys())
            s, e = sp["start"], sp["end"]
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < len(ans)
            assert ans[s : e + 1] == sp["text"]
            assert sp["type"] in valid_types
            assert sp["style"] in valid_styles
            assert 0.75 <= sp["risk"] <= 1.0
            assert 0.80 <= sp["confidence"] <= 1.0


def test_strict_cross_corpus_no_duplicates_and_non_confirmation(pairs: list[dict]) -> None:
    doc_hashes: set[str] = set()
    ans_hashes: set[str] = set()
    donor_hashes: set[str] = set()

    for pair in pairs:
        dh = sha256_text(pair["document_text"])
        ah = sha256_text(pair["model_output"])
        assert dh not in doc_hashes, f"Duplicate document in {pair['pair_id']}"
        assert ah not in ans_hashes, f"Duplicate answer in {pair['pair_id']}"
        doc_hashes.add(dh)
        ans_hashes.add(ah)

        base_corpus = pair["base_corpus"]
        assert base_corpus not in {"rushallu_rag", "sberquad_rag", "ruscibench_rag"}

        for sp in pair["spans"]:
            assert sp["donor_corpus"] != base_corpus, f"Same-corpus span in {pair['pair_id']}"
            sh = sha256_text(sp["text"])
            assert sh not in donor_hashes, f"Duplicate donor span in {pair['pair_id']}"
            donor_hashes.add(sh)

            assert sp["doc_stem_overlap"] <= 0.15
            assert sp["doc_ngram_overlap"] <= 0.25
            if sp["style"] == "short":
                assert sp["doc_stem_overlap"] == 0.0
            assert norm_text(sp["text"]) not in norm_text(pair["document_text"])


def test_zero_doc_overlap_with_rushallu_rag(pairs: list[dict], report: dict) -> None:
    blacklist = RusHalluBlacklist(DATASET_DIR)
    overlap = verify_zero_overlap(pairs, blacklist)
    assert overlap["doc_overlap"] == 0
    assert overlap["by_doc_id"] == 0
    assert overlap["by_full_doc_sha256"] == 0
    assert overlap["by_subdoc_sha256"] == 0
    assert overlap["by_prefix_100"] == 0
    assert overlap["by_word_8gram"] == 0
    assert overlap["by_inserted_clause"] == 0
    assert report["doc_overlap"] == 0


def test_style_distribution_and_oracle_ceilings(pairs: list[dict]) -> None:
    styles = Counter(sp["style"] for p in pairs for sp in p["spans"])
    total = sum(styles.values())
    assert total > 0
    # Realistic mix covering whole, clause, and short
    assert 0.25 <= styles["whole"] / total <= 0.55
    assert 0.35 <= styles["clause"] / total <= 0.65
    assert 0.05 <= styles["short"] / total <= 0.20

    summary = summarize_style_and_oracles(pairs, run_lexical_baseline=False)
    oracles = summary["oracle_ceiling"]
    assert oracles["oracle_candidate_top1"]["span_f1"] >= 0.60
    assert oracles["oracle_candidate_multi"]["span_f1"] >= 0.60
    assert oracles["oracle_sentence_multi"]["span_f1"] >= 0.60
    assert oracles["oracle_token_snap_expand"]["span_f1"] >= 0.60
