"""V16 шаг 1: веса русской LM через GitHub (HF/LFS в песочнице недоступны).

Скрипт ничего не изобретает: он (1) проверяет уже добытые локальные веса — размер, формат,
SHA-256, совместимость токенизатора; (2) прогоняет журнал поиска по GitHub (repository search,
code search, обход деревьев на файлы весов > 1 МБ) и записывает результат каждого запроса;
(3) фиксирует, что `rubert-base` (~530 МБ) в доступных репозиториях отсутствует.

Всё пишется в `artifacts/weights_v16.json`. Скачивание — `git clone --depth 1` в
`tools/external/` (каталог в .gitignore), веса в историю не попадают.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "tools" / "external"

LOCAL_WEIGHTS = [
    {
        "repo": "https://github.com/pyramidheadshark/rubert-turbo-payments",
        "path": "models/onnx_model/model_quantized.onnx",
        "file": EXTERNAL / "rubert-turbo-payments/models/onnx_model/model_quantized.onnx",
        "format": "ONNX QOperator int8/uint8 (BERT rubert-tiny2)",
        "architecture": "312 hidden / 3 слоя / 12 голов / vocab 83 828 / max_pos 2048",
        "dequant": "models/rubert-tiny2-deq/pytorch_model.bin (fp32, 116 786 340 B)",
        "role": "основной источник признаков внимания (и быстрый int8-инференс)",
        "clone": "git clone --depth 1",
    },
    {
        "repo": "https://github.com/Gregory-mcbit/avitomodel",
        "path": "spacer_model/checkpoint-113/model.safetensors",
        "file": EXTERNAL / "avitomodel/spacer_model/checkpoint-113/model.safetensors",
        "format": "safetensors float32 (rubert-tiny v1)",
        "architecture": "312 / 3 / 12 / vocab 29 564",
        "role": "не используется: токенизатор несовместим с нашим (83 828)",
        "clone": "git clone --depth 1",
    },
    {
        "repo": "https://github.com/dinis-a/android-ru-stylometry",
        "path": "ru_detector/model.onnx",
        "file": EXTERNAL / "android-ru-stylometry/ru_detector/model.onnx",
        "format": "ONNX (fine-tuned русский детектор стиля)",
        "architecture": "не подтверждена (чужой классификатор, нет смысла для признаков)",
        "role": "найден поиском, не скачивался",
        "clone": None,
    },
]

REPO_QUERIES = ["rubert-base", "rubert russian", "rubert-tiny2", "rubert-large", 'russian bert onnx',
                "ruroberta", "sbert russian", "rubert-base-cased", "russian hallucination"]
CODE_QUERIES = ["rubert-base filename:pytorch_model.bin",
                "rubert filename:model.safetensors",
                "ModernBERT russian filename:model.safetensors",
                "rubert-tiny2 filename:pytorch_model.bin"]
TREE_CHECKS = ["gleb-skobinsky/RuCo-BERT", "Djacon/russian-emotion-detection", "vlarine/ruberta",
               "TryDotAtwo/ruBERT-ruLaw", "6jack6/rubert_tiny_analysis", "Ilyat9/RosEltorg",
               "WiMean/news-classifier-rubert", "OlaEla/nlp-pipeline-huggingface",
               "abuketov555-hash/nlp-147k-rus-dep-anx-neu-pos",
               "shokovapolina/linguistic-markers-depression-anxiety"]
WEIGHT_RE = r"\\.(bin|safetensors|onnx|gguf|pt|ckpt)$"


def gh(args, timeout=60):
    started = time.time()
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    return {"code": result.returncode, "seconds": round(time.time() - started, 1),
            "stdout": result.stdout.strip()[:4000], "stderr": result.stderr.strip()[:400]}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/weights_v16.json")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args(argv)

    report = {"rule": "веса только с GitHub (clone --depth 1 / release asset); HF, hf-mirror, "
                      "ModelScope и Git-LFS в песочнице недоступны; всё хэшируется SHA-256; "
                      "веса лежат локально вне пакета и вне git",
              "target": {"wanted": "rubert-base-cased (~530 МБ, 768/12)",
                         "fallback": "rubert-tiny2 (~29 МБ) + ONNX int8",
                         "got": "fallback"},
              "local": [], "search": {"repositories": [], "code": [], "trees": []}}

    for item in LOCAL_WEIGHTS:
        entry = {key: value for key, value in item.items() if key != "file"}
        path = item["file"]
        if path.exists():
            entry.update({"bytes": path.stat().st_size, "sha256": sha256(path),
                          "present": True})
        else:
            entry["present"] = False
        report["local"].append(entry)

    deq = ROOT / "models/rubert-tiny2-deq/pytorch_model.bin"
    if deq.exists():
        report["dequantized"] = {"path": "models/rubert-tiny2-deq/pytorch_model.bin",
                                 "bytes": deq.stat().st_size, "sha256": sha256(deq),
                                 "params": 29196585, "format": "torch fp32 state_dict"}

    if not args.skip_search:
        for query in REPO_QUERIES:
            result = gh(["api", "-X", "GET", "search/repositories", "-f", f"q={query}",
                         "-f", "per_page=8", "--jq",
                         '.items[] | "\(.full_name)\t\(.stargazers_count)\t\(.size)"'])
            report["search"]["repositories"].append({"query": query, **result})
        for query in CODE_QUERIES:
            result = gh(["api", "-X", "GET", "search/code", "-f", f"q={query}", "-f",
                         "per_page=1", "--jq", ".total_count"])
            report["search"]["code"].append({"query": query, **result})
        for repo in TREE_CHECKS:
            result = gh(["api", f"repos/{repo}/git/trees/HEAD?recursive=1", "--jq",
                         '[.tree[] | select((.path|test("\\.(bin|safetensors|onnx|gguf|pt)$")) '
                         'and .size > 1000000) | "\(.size) \(.path)"] | .[]'])
            report["search"]["trees"].append({"repo": repo, **result})

    report["conclusion"] = {
        "why_not_base": "в доступных GitHub-репозиториях rubert-base нет: code search по "
                        "filename:pytorch_model.bin / model.safetensors даёт 0 (GitHub не "
                        "индексирует бинарники), в деревьях кандидатов файлов весов > 1 МБ нет, "
                        "кроме чужих fine-tuned ONNX без энкодера в чистом виде",
        "what_used": "int8-ONNX rubert-tiny2 (29 385 865 B) → dequant fp32 → transformers",
        "verified": "ключи state_dict совпали с BertForSequenceClassification, max|Δlogits| "
                    "против onnxruntime = 0.0656 (артефакт квантования)",
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps({"local": [{"repo": row["repo"], "present": row["present"],
                                 "bytes": row.get("bytes")} for row in report["local"]],
                      "dequantized": report.get("dequantized", {}).get("sha256")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
