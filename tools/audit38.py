import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ["tests/test_base33_safe.py", "tests/test_base33_training.py", "tests/test_preflight_13.py", "tests/test_sweep_ragtruth30.py"]
CONTEXTS = [("prompt33", "a440d9e", BASE), ("prompt34", "a0929fd", BASE), ("prompt36", "aeef1a3", BASE + ["tests/test_nested36.py"]), ("prompt37_original", "b649d06", BASE + ["tests/test_nested36.py", "tests/test_nested37.py"]), ("prompt37_fixed", "803f57a", BASE + ["tests/test_nested36.py", "tests/test_nested37.py"])]


def main():
    target = ROOT / "artifacts/nested38_test_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    output = {"status": "в процессе", "tests_use_tee": False, "test301": False, "contexts": []}
    for name, commit, tests in CONTEXTS:
        print(f"Проверка {name} на {commit} без tee", flush=True)
        raw = subprocess.check_output(["git", "archive", commit, "spanver", "tools", "tests", "scripts", "conftest.py"], cwd=ROOT)
        with tempfile.TemporaryDirectory(prefix="audit38-") as folder:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
                for member in archive.getmembers():
                    if member.name.startswith("/") or ".." in Path(member.name).parts or member.issym() or member.islnk():
                        raise ValueError("небезопасный путь в архиве git")
                archive.extractall(folder)
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=folder, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        output["contexts"].append({"name": name, "commit": commit, "tests": tests, "exit_code": result.returncode, "output": result.stdout})
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    expected = {"prompt33": 0, "prompt34": 0, "prompt36": 0, "prompt37_original": 1, "prompt37_fixed": 0}
    output["status"] = "готово" if all(row["exit_code"] == expected[row["name"]] for row in output["contexts"]) else "STOP: результаты отличаются от ожидаемых"
    output["historical_36"] = "пять сохранённых журналов: 19 passed, 0 failed в каждом; tee не скрыл падений"
    output["historical_37"] = "пять сохранённых журналов: 22 passed, 1 failed в каждом; один и тот же тест скрыт tee"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if output["status"] == "готово" else 2


if __name__ == "__main__":
    raise SystemExit(main())
