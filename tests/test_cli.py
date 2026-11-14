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
