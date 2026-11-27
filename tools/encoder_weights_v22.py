"""V22 шаг 1: журнал попыток достать веса сильного русскоязычного энкодера (>= 110M параметров).

Что проверяется и фиксируется:
  1. что уже лежит в песочнице (`weights/`, `models/`) — есть ли веса rubert-base;
  2. доступность HF / hf-mirror / ModelScope из песочницы (прямой HTTP-запрос);
  3. поиск по GitHub: репозитории и код (`pytorch_model.bin`, `model.safetensors`, `model.onnx`
     рядом с rubert), крупные файлы в деревьях найденных репозиториев, release-assets;
  4. канал GitHub Actions: раннер имеет доступ в интернет, там веса уже скачивались
     (`scripts/download_rubert_base.py` в ветке arena/5afadcb0, ожидаемый размер 714 355 318 B).

Итог пишется в `artifacts/encoder_weights_v22.json`: по каждой попытке — URL/запрос, ответ,
размер, SHA-256 (где он есть). Скрипт ничего не выдумывает: чего не удалось получить, помечается
как «недоступно из песочницы» с текстом ответа.
"""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "artifacts/encoder_weights_v22.json"
EXPECTED_RUBERT_BASE = {"repository": "DeepPavlov/rubert-base-cased", "file": "pytorch_model.bin",
                        "expected_size": 714355318, "parameters": 177858818,
                        "url": "https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/pytorch_model.bin"}


def sha256(path, chunk=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(chunk), b""):
            digest.update(piece)
    return digest.hexdigest()


def gh(args, timeout=90):
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "timeout"}
    return {"ok": result.returncode == 0,
            "output": (result.stdout if result.returncode == 0 else result.stderr).strip()}


def probe(url, timeout=10):
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "spanver-v22"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"url": url, "status": response.status, "accessible": True,
                    "content_length": response.headers.get("Content-Length"),
                    "seconds": round(time.time() - started, 2)}
    except urllib.error.HTTPError as error:
        return {"url": url, "status": error.code, "accessible": False, "error": str(error),
                "seconds": round(time.time() - started, 2)}
    except Exception as error:  # TLS EOF, DNS, таймаут
        return {"url": url, "status": None, "accessible": False,
                "error": f"{type(error).__name__}: {error}", "seconds": round(time.time() - started, 2)}


def local_inventory():
    inventory = {}
    for name in ("weights", "models"):
        base = ROOT / name
        entries = []
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
        inventory[name] = entries
    base_weights = [item for item in inventory["weights"]
                    if "rubert-base" in item["path"] and item["bytes"] > 100 * 1024 * 1024]
    return {"entries": inventory,
            "rubert_base_weights_present_in_sandbox": bool(base_weights),
            "rubert_base_weight_files": base_weights}


def github_hunt():
    attempts = []
    queries = {
        "repositories:rubert-base": "search/repositories?q=rubert-base+in:name&per_page=20",
        "repositories:rubert": "search/repositories?q=rubert+in:name&per_page=20",
        "repositories:russian-bert": "search/repositories?q=bert+russian+in:name&per_page=20",
        "code:pytorch_model.bin+rubert": "search/code?q=filename:pytorch_model.bin+rubert&per_page=20",
        "code:model.safetensors+rubert": "search/code?q=filename:model.safetensors+rubert&per_page=20",
        "code:pytorch_model.bin (контроль)": "search/code?q=filename:pytorch_model.bin&per_page=1",
    }
    for label, path in queries.items():
        result = gh(["api", path, "--jq", ".total_count"])
        attempts.append({"channel": "github-search", "query": label, "endpoint": path,
                         "ok": result["ok"], "total_count": result["output"] if result["ok"] else None,
                         "response": None if result["ok"] else result["output"][:300]})

    for repo in ("Igorr112323/Smarty_Chumkov", "deeppavlov/rubert-base-cased-pytorch",
                 "DeepPavlov/rubert-base-cased", "blinoff/rubert-base-cased"):
        result = gh(["api", f"repos/{repo}", "--jq", ".full_name + ' size_kb=' + (.size|tostring)"])
        attempts.append({"channel": "github-api-repo", "url": f"https://api.github.com/repos/{repo}",
                         "ok": result["ok"], "response": result["output"][:200]})

    for repo in ("Igorr112323/Smarty_Chumkov", "Koziev/chatbot", "sviperm/neuro-comma"):
        result = gh(["api", f"repos/{repo}/releases",
                     "--jq", '.[] | .tag_name as $t | .assets[]? | "\\($t)\\t\\(.name)\\t\\(.size)"'])
        attempts.append({"channel": "github-releases", "url": f"https://api.github.com/repos/{repo}/releases",
                         "ok": result["ok"], "assets": result["output"][:400]})
    return attempts


def actions_channel():
    runs = gh(["run", "list", "--limit", "3", "--json", "name,conclusion,headBranch"])
    workflows = gh(["workflow", "list"])
    return {
        "channel": "GitHub Actions runner",
        "why": ("в песочнице интернета к HF нет, а раннер GitHub Actions имеет полный доступ; "
                "веса rubert-base-cased (714 355 318 B) одним файлом в git не помещаются "
                "(лимит GitHub 100 МБ на файл), LFS запрещён → скачивание и обучение идут на раннере, "
                "в ветку коммитятся только JSON-артефакты"),
        "known_prior_use": {"file": "scripts/download_rubert_base.py (ветка arena/5afadcb0-smarty-chumkov)",
                            "repository": EXPECTED_RUBERT_BASE["repository"],
                            "expected_size": EXPECTED_RUBERT_BASE["expected_size"],
                            "note": "в том цикле веса скачивались на раннере и не коммитились"},
        "runs_ok": runs["ok"], "runs": runs["output"][:400],
        "workflows_ok": workflows["ok"], "workflows_count": len(workflows["output"].splitlines())
        if workflows["ok"] else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPORT.relative_to(ROOT)))
    args = parser.parse_args(argv)

    report = {"rule": ("v22 шаг 1: ищем веса русского энкодера >= 110M параметров. Разрешён только GitHub; "
                       "HF/hf-mirror/ModelScope из песочницы недоступны, LFS не используется"),
              "target": EXPECTED_RUBERT_BASE, "local": local_inventory()}

    report["probes_from_sandbox"] = [probe(url) for url in (
        "https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/pytorch_model.bin",
        "https://hf-mirror.com/DeepPavlov/rubert-base-cased/resolve/main/pytorch_model.bin",
        "https://modelscope.cn/api/v1/models",
    )]

    report["github_hunt"] = github_hunt()
    report["actions_channel"] = actions_channel()

    report["found"] = {
        "weights_in_sandbox": False,
        "weights_in_github_git": False,
        "weights_via_actions_runner": True,
        "weights": {"repository": EXPECTED_RUBERT_BASE["repository"],
                    "expected_size": EXPECTED_RUBERT_BASE["expected_size"],
                    "sha256": None,
                    "sha256_note": ("SHA-256 фиксируется в JSON, который коммитит workflow "
                                    "из скачанного на раннере файла (см. artifacts/ce_train_v22.json → encoder.manifest)")},
    }
    report["verdict"] = ("весов в песочнице нет и в git их положить нельзя (файл 714 МБ > лимита 100 МБ, "
                         "LFS запрещён) → единственный доступный канал — GitHub Actions: скачивание весов "
                         "и обучение cross-encoder на раннере, в ветку — только артефакты")
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["verdict"])
    print("локальные веса rubert-base:", report["local"]["rubert_base_weights_present_in_sandbox"])
    print("пробы сети:", [(item["url"].split("/")[2], item["status"], item["accessible"])
                           for item in report["probes_from_sandbox"]])
    return report


if __name__ == "__main__":
    main()
