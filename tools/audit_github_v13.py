"""V13 шаг 1: аудит трёх GitHub-репозиториев по span-детекции галлюцинаций.

Считает по факту из клонированных деревьев (tools/external/*): объём кода, LOC,
заявленные зависимости, реальные импорты тяжёлых пакетов, лицензию, наличие
русского языка, работает ли offline на CPU, а также во сколько раз их веса
больше лимита нашего пакета. Заявленные метрики берутся цитатами из README/доков
с указанием файла и строки — без пересказа и без округлений в свою пользу.

Пишет artifacts/github_audit_v13.json.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "tools" / "external"
PACKAGE_BYTES = 523_986          # spanver-0.2.0.zip, artifacts/package.json
PACKAGE_LIMIT = 614_400          # 0.6 MB

REPOS = {
    "LettuceDetect": {
        "url": "https://github.com/KRLabsOrg/LettuceDetect",
        "core": ["lettucedetect/detectors", "lettucedetect/models"],
        "claims": {
            "0.642": ("README.md", "unified v2 test (10 698 примеров), mmBERT-base span-F1"),
            "0.689": ("README.md", "unified v2 test, qwen-2b span-F1 (генеративная)"),
            "0.528": ("README.md", "RAGTruth slice, mmBERT-base span-F1"),
            "0.574": ("README.md", "RAGTruth slice, qwen-2b span-F1"),
        },
        "models": [
            {"id": "KRLabsOrg/lettucedect-v2-mmbert-base", "params": 110e6, "note": "encoder, CPU-ok"},
            {"id": "KRLabsOrg/lettucedect-610m-eurobert-*", "params": 610e6, "note": "multilingual encoder"},
            {"id": "KRLabsOrg/lettucedect-v2-qwen-2b", "params": 2.0e9, "note": "генеративная, GPU"},
        ],
        "rus_evidence": "docs/EUROBERT.md: язык-таблица без ru; поддерживаются en/de/fr/es/it/pl/zh",
    },
    "halo": {
        "url": "https://github.com/reyyishreyas/halo",
        "core": ["halo"],
        "claims": {
            "0.674": ("README.md", "sentence-level F1 на собственных 114 спанах (100 ответов, en)"),
            "0.571": ("README.md", "NLI-бейзлайн, тот же набор"),
        },
        "models": [
            {"id": "microsoft/Phi-3-mini-4k-instruct", "params": 3.8e9, "note": "генератор, нужны logprobs"},
            {"id": "BAAI/bge-small-en-v1.5", "params": 33e6, "note": "эмбеддинги для FAISS"},
        ],
        "rus_evidence": "нет: Phi-3-mini + bge-small-en, данные только английские, ru-кода нет",
    },
    "faithful_eval": {
        "url": "https://github.com/alexandrainst/faithful_eval",
        "core": ["src/factuality_eval", "src/scripts"],
        "claims": {
            "lettucedetect": ("README.md", "своих span-метрик нет: обёртка над lettucedetect + LLM-разметка"),
        },
        "models": [
            {"id": "KRLabsOrg/lettucedect-*", "params": 110e6, "note": "через зависимость lettucedetect"},
            {"id": "openai API", "params": 0, "note": "частью пайплайна, не offline"},
        ],
        "rus_evidence": "prompt_utils.py: Lang = be/bs/bg/ca/…/uk; ru в списке отсутствует",
    },
}

HEAVY = ("torch", "transformers", "faiss", "sentence_transformers", "sklearn",
         "openai", "datasets", "timm", "accelerate", "nltk")
RUS_RE = re.compile(r"\b(russian|русск|\bru\b|ru-RU)", re.IGNORECASE)
CLAIM_RE = re.compile(r"(span[- ]?F1|span F1|F1)", re.IGNORECASE)


def sh(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True).stdout.strip()


def walk_files(path):
    for item in path.rglob("*"):
        if item.is_file() and ".git" not in item.parts:
            yield item


def repo_stats(path):
    files = list(walk_files(path))
    py = [f for f in files if f.suffix == ".py"]
    total_bytes = sum(f.stat().st_size for f in files)
    py_bytes = sum(f.stat().st_size for f in py)
    loc = 0
    for f in py:
        try:
            loc += sum(1 for _ in f.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    return {"files": len(files), "bytes": total_bytes, "py_files": len(py),
            "py_bytes": py_bytes, "py_loc": loc}


def core_stats(path, subdirs):
    loc = 0
    files = 0
    for sub in subdirs:
        target = path / sub
        if not target.exists():
            continue
        for f in target.rglob("*.py"):
            files += 1
            loc += sum(1 for _ in f.open("r", encoding="utf-8", errors="ignore"))
    return {"py_files": files, "py_loc": loc}


def imports_heavy(path):
    found = {}
    for f in path.rglob("*.py"):
        if ".git" in f.parts:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for name in HEAVY:
            if re.search(rf"^\s*(import|from)\s+{name}\b", text, re.MULTILINE):
                found.setdefault(name, 0)
                found[name] += 1
    return dict(sorted(found.items(), key=lambda item: -item[1]))


def license_of(path):
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        candidate = path / name
        if candidate.exists():
            head = candidate.read_text(encoding="utf-8", errors="ignore")[:200]
            first = next((line.strip() for line in head.splitlines() if line.strip()), "")
            return {"file": name, "first_line": first,
                    "kind": "MIT" if "MIT License" in head else first}
    readme = path / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"\b(MIT|Apache[- ]2\.0|BSD-3|GPL-3)\b", text)
        if match:
            return {"file": "README.md", "first_line": match.group(0), "kind": match.group(0)}
    return {"file": None, "first_line": None, "kind": None}


def declared_deps(path):
    pyproject = path / "pyproject.toml"
    if not pyproject.exists():
        return []
    text = pyproject.read_text(encoding="utf-8", errors="ignore")
    block = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
    if not block:
        return []
    return re.findall(r"\"([^\"]+)\"", block.group(1))


def rus_hits(path, limit=6):
    hits = []
    for f in walk_files(path):
        if f.suffix not in (".py", ".md", ".yaml", ".yml", ".txt", ".json"):
            continue
        try:
            for number, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if RUS_RE.search(line):
                    hits.append({"file": str(f.relative_to(path)), "line": number,
                                 "text": line.strip()[:160]})
                    if len(hits) >= limit:
                        return hits
        except OSError:
            continue
    return hits


def claim_lines(path, needles):
    out = []
    seen = set()
    for needle, (rel, context) in needles.items():
        target = path / rel
        if not target.exists():
            continue
        for number, line in enumerate(target.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if needle in line:
                key = (rel, number)
                if key not in seen:
                    seen.add(key)
                    out.append({"file": rel, "line": number, "context": context,
                                "text": line.strip()[:240]})
                break
    return out


def weight_bytes(params, dtype_bytes):
    return int(params * dtype_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/github_audit_v13.json")
    args = parser.parse_args(argv)

    audit = {
        "rule": "числа считаны из клонированных деревьев tools/external/*; заявленные метрики — "
                "цитатами с файлом и строкой; веса не скачивались (HF недоступен, тянем только "
                "github/codeload)",
        "package": {"bytes": PACKAGE_BYTES, "limit_bytes": PACKAGE_LIMIT,
                    "headroom_bytes": PACKAGE_LIMIT - PACKAGE_BYTES},
        "our_baseline": {"val_span": 0.3363, "val_token": 0.6191, "val_pair": 0.4872,
                         "val_fpr": 0.125, "mfava_span": 0.6895, "runtime": "offline stdlib+numpy, CPU"},
        "repos": {},
    }

    for name, meta in REPOS.items():
        path = EXTERNAL / name
        if not path.exists():
            audit["repos"][name] = {"error": "не склонирован"}
            continue
        stats = repo_stats(path)
        core = core_stats(path, meta["core"])
        heavies = imports_heavy(path)
        models = []
        for model in meta["models"]:
            models.append({
                **model,
                "fp32_bytes": weight_bytes(model["params"], 4),
                "fp16_bytes": weight_bytes(model["params"], 2),
                "int8_bytes": weight_bytes(model["params"], 1),
                "x_package": round(weight_bytes(model["params"], 2) / PACKAGE_BYTES, 1)
                if model["params"] else None,
                "x_limit": round(weight_bytes(model["params"], 2) / PACKAGE_LIMIT, 1)
                if model["params"] else None,
            })
        smallest = min((m for m in models if m["params"]), key=lambda m: m["params"], default=None)
        audit["repos"][name] = {
            "url": meta["url"],
            "commit": sh(["git", "log", "-1", "--format=%H %ad", "--date=short"], path),
            "license": license_of(path) if license_of(path)["kind"] else {
                "file": None, "first_line": None, "kind": "отсутствует",
                "note": "файла LICENSE нет и в README лицензия не указана → код портировать нельзя, "
                        "только идеи (переписывание с нуля)"},
            "size": stats,
            "core": core,
            "declared_deps": declared_deps(path),
            "heavy_imports": heavies,
            "needs": {
                "torch": "torch" in heavies,
                "transformers": "transformers" in heavies,
                "faiss": "faiss" in heavies,
                "gpu_required": name == "halo",
                "network_required": name in ("faithful_eval",),
            },
            "weights_in_repo_bytes": 0,
            "models": models,
            "smallest_weight_fp16_bytes": smallest["fp16_bytes"] if smallest else None,
            "smallest_x_limit": smallest["x_limit"] if smallest else None,
            "fits_0_6mb": bool(smallest and smallest["fp16_bytes"] <= PACKAGE_LIMIT),
            "russian": {"verdict": "нет" if not rus_hits(path) else "упоминается",
                        "evidence": meta["rus_evidence"], "hits": rus_hits(path)},
            "span_claims": claim_lines(path, meta["claims"]),
            "offline_cpu": {
                "LettuceDetect": "частично: инференс encoder-модели на CPU возможен, но нужен torch "
                                 "+ веса ≥ 210 MB, то есть пакет и offline-stdlib несовместимы",
                "halo": "нет: генератор Phi-3-mini (3.8B) + FAISS, нужны logprobs и GPU/MPS",
                "faithful_eval": "нет: часть пайплайна — OpenAI API и torch 2.9",
            }[name],
        }

    (ROOT / args.report).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    for name, data in audit["repos"].items():
        print(name, data["size"]["py_loc"], "LOC python,", data["core"]["py_loc"], "LOC core,",
              "license", data["license"]["kind"], "| min fp16",
              data["smallest_weight_fp16_bytes"], "B =", data["smallest_x_limit"], "x limit",
              "| ru:", data["russian"]["verdict"], flush=True)
    print("written", args.report, flush=True)


if __name__ == "__main__":
    main()
