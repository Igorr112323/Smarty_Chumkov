import json
import subprocess
import sys
from pathlib import Path

from spanver import ranker
from spanver.module import verify

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "spanver" / "ranker_weights.json"


def test_weights_are_bundled_and_valid():
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    assert len(weights["cand"]["w"]) == ranker.FEATURE_COUNT
    assert len(weights["pair"]["w"]) == ranker.PAIR_FEATURE_COUNT
    assert len(weights["mode"]["w"]) == ranker.MODE_FEATURE_COUNT
    assert weights["local_hgb"]["trees"]
    assert len(weights["mode"]["w"]) == len(weights["mode"]["mu"]) == len(weights["mode"]["sd"])
    assert 0.0 <= weights["config"]["gate"] <= 1.0
    assert 0.0 <= weights["config"]["threshold"] <= 1.0
    assert weights["config"]["max_spans"] >= 1


def test_default_verify_uses_ranker_without_third_party_imports():
    code = (
        "import sys; import spanver.module as m; "
        "r=m.verify('В документе написано 45 процентов. Комета — это Marketing.', '45 процентов.'); "
        "print(r['ai_share']); assert 'sklearn' not in sys.modules and 'numpy' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 0.0 <= float(result.stdout.strip()) <= 1.0


def test_ranker_output_is_deterministic_and_bounded():
    answer = "В 1990 году завод выпустил 1200 деталей. Комета — это Marketing."
    document = "Завод выпускает детали с 1990 года."
    first = verify(answer, document)
    second = verify(answer, document)
    assert first["spans"] == second["spans"]
    assert first["ai_share"] == second["ai_share"]
    assert first["meta"]["n_spans"] == second["meta"]["n_spans"]
    for span in first["spans"]:
        assert 0 <= span["start"] <= span["end"] < len(answer)
        assert 0.0 <= span["risk"] <= 1.0
        assert 0.0 <= span["confidence"] <= 1.0
    assert len(first["spans"]) <= ranker._model()["config"]["max_spans"]


def test_explicit_threshold_disables_ranker():
    answer = "В 1990 году завод выпустил 1200 деталей. Возможно, это неверно."
    document = "Завод."
    ranked = verify(answer, document)
    legacy = verify(answer, document, threshold=0.98, use_ranker=False)
    assert set(legacy) == {"spans", "ai_share", "meta"}
    assert all(span["risk"] >= 0.98 for span in legacy["spans"])
    assert set(ranked) == set(legacy)


def test_long_local_pair_does_not_select_whole_answer():
    path = ROOT / "artifacts" / "mfava_train_v2.jsonl"
    if not path.exists():
        return
    pair = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    answer = pair["model_output"]
    spans = ranker.select_spans(answer, pair["document_text"])
    assert spans, "локальный каскад должен находить внутрипредложные фрагменты"
    assert len(spans) <= ranker._model()["local_config"]["max_spans"]
    assert not any(span["start"] == 0 and span["end"] == len(answer) - 1 for span in spans)
