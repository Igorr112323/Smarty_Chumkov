import json
import subprocess
import sys
from pathlib import Path

from spanver.module import CALIBRATION_POINTS, calibrate_share, sentence_spans, verify

ROOT = Path(__file__).resolve().parent.parent
MIXTURE = ROOT / "artifacts" / "calib_mixture.jsonl"
TOOL = ROOT / "tools" / "make_calibration.py"


def ensure_mixtures():
    if not MIXTURE.exists():
        subprocess.run([sys.executable, str(TOOL)], cwd=ROOT, check=True, capture_output=True)


def read_mixtures():
    ensure_mixtures()
    with open(MIXTURE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_mixtures_present_and_ordered():
    rows = read_mixtures()
    assert len(rows) == 5
    assert [row["share"] for row in rows] == [0.0, 0.25, 0.5, 0.75, 1.0]
    for row in rows:
        assert row["n_human"] + row["n_ai"] == 40
        assert row["chars"] > 0
        assert row["document"]


def test_mixture_regeneration_is_byte_identical():
    ensure_mixtures()
    before = MIXTURE.read_bytes()
    subprocess.run([sys.executable, str(TOOL)], cwd=ROOT, check=True, capture_output=True)
    assert MIXTURE.read_bytes() == before


def test_calibrate_share_is_monotone_and_bounded():
    values = [calibrate_share(raw) for raw, _ in CALIBRATION_POINTS]
    assert values == sorted(values)
    assert calibrate_share(0.0) == 0.0
    assert calibrate_share(1.0) == 1.0
    assert 0.0 <= calibrate_share(0.42) <= 1.0


def test_ai_share_tracks_share_and_stays_in_range():
    measured = []
    for row in read_mixtures():
        value = verify(row["text"], row["document"])["ai_share"]
        assert 0.0 <= value <= 1.0
        assert abs(value - row["share"]) <= 0.05
        measured.append(value)
    assert measured == sorted(measured)


def test_sentence_spans_cover_answer():
    text = "Первая фраза здесь. Вторая фраза тоже здесь!"
    spans = sentence_spans(text)
    assert spans
    for start, end in spans:
        assert 0 <= start <= end < len(text)
