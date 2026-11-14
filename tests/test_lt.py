from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import build_tiny

from spanver.lt import BaseDetector, TransformerDetector

ROOT = Path(__file__).resolve().parent.parent
FILES = ("base.py", "transformer.py", "trainer.py", "evaluator.py")
COMMIT = "dcce161ff7c8a46147cae5f0bda14118bef7a2d6"
ANSWER = "Москва основана в 1990 году князем Юрием."


class ForcePositive(torch.nn.Module):
    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 2)
        logits[:, :, 1] = 5.0
        return type("Output", (), {"logits": logits})()


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lt_encoder")
    build_tiny(directory, token_classification=True)
    return TransformerDetector(str(directory), max_length=512, device="cpu", lang="ru")


def test_vendored_files_keep_copyright():
    for name in FILES:
        text = (ROOT / "spanver" / "lt" / name).read_text(encoding="utf-8")
        assert "Copyright (c) 2025 KR Labs" in text
        assert COMMIT in text
        assert "MIT" in text


def test_third_party_notice_exists():
    text = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
    assert "LettuceDetect" in text
    assert COMMIT in text
    assert "MIT" in text


def test_detector_returns_list(detector):
    spans = detector.predict(
        context=["Москва — древний город."], question="Когда основана Москва?", answer=ANSWER, output_format="spans"
    )
    assert isinstance(spans, list)


def test_forced_span_has_character_offsets(detector):
    detector.model = ForcePositive()
    spans = detector.predict(context=["Москва — древний город."], question="Когда?", answer=ANSWER, output_format="spans")
    assert spans
    for span in spans:
        assert 0 <= span["start"] <= span["end"] <= len(ANSWER)
        assert span["text"] == ANSWER[span["start"] : span["end"]]
    assert spans[0]["start"] == 0


def test_base_detector_is_abstract():
    assert issubclass(TransformerDetector, BaseDetector)
