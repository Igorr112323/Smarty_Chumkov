import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cli(args):
    return subprocess.run([sys.executable, "-m", "spanver"] + args, capture_output=True, text=True, cwd=ROOT)


def test_verify_cli_prints_json():
    result = run_cli(
        ["verify", "--answer", "В 1990 году завод выпустил 1200 деталей.", "--document", "Завод выпускает детали."]
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"spans", "ai_share", "meta"}


def test_verify_cli_reads_files(tmp_path):
    answer = tmp_path / "answer.txt"
    document = tmp_path / "document.txt"
    answer.write_text("В 1990 году завод выпустил 1200 деталей.", encoding="utf-8")
    document.write_text("Завод выпускает детали.", encoding="utf-8")
    result = run_cli(["verify", "--answer", str(answer), "--document", str(document)])
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["meta"]["answer_len"] == 40


def test_model_backend_without_weights_prints_message():
    result = run_cli(["verify", "--backend", "model", "--weights", "models/none", "--answer", "а", "--document", "б"])
    assert result.returncode == 2
    assert "веса не найдены" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_model_backend_without_head_prints_message(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    (weights / "model.safetensors").write_bytes(b"stub")
    result = run_cli(
        ["verify", "--backend", "model", "--weights", str(weights), "--head", str(tmp_path / "none.json"), "--answer", "а", "--document", "б"]
    )
    assert result.returncode == 2
    assert "веса головки не найдены" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr
