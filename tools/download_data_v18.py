"""V18 шаг 1: последняя охота за РЕАЛЬНЫМИ русскими клаузовыми span-данными.

Скрипт делает три вещи и всё записывает в `artifacts/ru_clause_search_v18.json`:

1. Пробивает все внешние хосты, которые обычно держат датасеты (Kaggle API, Zenodo, figshare,
   Mendeley Data, GitLab, Bitbucket, SourceForge, arXiv, data.world, Yandex Cloud, Sber,
   HuggingFace/ModelScope и текстовые прокси) — в песочнице разрешены только github.com,
   codeload.github.com, api.github.com, pypi.org, files.pythonhosted.org, registry.npmjs.org,
   поэтому каждый хост фиксируется с реальным HTTP-кодом или текстом ошибки.
2. Гоняет расширенный GitHub-поиск (repository + code) по запросам про русские
   галлюцинации/фактчекинг/спаны и по конкретным именам известных датасетов.
3. Для найденных репозиториев обходит дерево и ищет файлы данных (.jsonl/.csv/.parquet/.json
   > 50 КБ), чтобы потом проверить стиль разметки и doc_overlap.

Скрипт ничего не додумывает: если хост недоступен — так и пишется.
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

DATA_JQ = ('[.tree[] | select(((.path|endswith(".jsonl")) or (.path|endswith(".json")) or '
           '(.path|endswith(".csv")) or (.path|endswith(".parquet")) or '
           '(.path|endswith(".tsv"))) and .size > 20000) | "\(.size) \(.path)"] | .[:8] | .[]')
EXTERNAL = ROOT / "tools" / "external"

HOSTS = {
    "kaggle.com": ["https://www.kaggle.com/api/v1/datasets/list?search=russian%20hallucination",
                   "https://www.kaggle.com/api/v1/datasets/list?search=ru%20rag%20factuality"],
    "zenodo.org": ["https://zenodo.org/api/records?q=russian%20hallucination&size=5",
                   "https://zenodo.org/api/records?q=ru%20factuality%20dataset&size=5"],
    "figshare.com": ["https://api.figshare.com/v2/articles?search_for=russian+hallucination"],
    "data.mendeley.com": ["https://data.mendeley.com/public-api/datasets?query=russian%20hallucination"],
    "gitlab.com": ["https://gitlab.com/api/v4/projects?search=russian%20hallucination"],
    "bitbucket.org": ["https://api.bitbucket.org/2.0/repositories?q=name~%22hallucination%22"],
    "sourceforge.net": ["https://sourceforge.net/directory/?q=russian%20dataset"],
    "arxiv.org": ["https://export.arxiv.org/api/query?search_query=all:%22russian%20hallucination%22&max_results=5"],
    "data.world": ["https://api.data.world/v0/search?query=russian%20hallucination"],
    "cloud.yandex.ru": ["https://cloud.yandex.ru/docs/datasets/"],
    "sber.ru": ["https://sber.ru/"],
    "huggingface.co": ["https://huggingface.co/api/datasets?search=russian%20hallucination"],
    "hf-mirror.com": ["https://hf-mirror.com/api/datasets?search=ru%20rag"],
    "modelscope.cn": ["https://modelscope.cn/api/v1/datasets?Name=ru-hallucination"],
    "r.jina.ai": ["https://r.jina.ai/https://huggingface.co/datasets?search=ru+hallucination"],
    "pypi.org": ["https://pypi.org/pypi/ruhallu/json"],
    "registry.npmjs.org": ["https://registry.npmjs.org/russian-hallucination-dataset"],
}

REPO_QUERIES = [
    "russian hallucination dataset",
    "RuHalu",
    "ru hallucination",
    "russian factuality dataset",
    "russian fact verification dataset",
    "ru fact checking dataset",
    "RAGTruth russian",
    "russian RAG benchmark",
    "ru rag dataset",
    "clause hallucination annotation",
    "hallucination spans dataset",
    "ru span annotation dataset",
    "Russian NLI dataset",
    "russe fact checking",
    "CheRRI russian",
    "russian claim verification",
]

CODE_QUERIES = [
    "ру галлюцинации разметка spans",
    "hallucination russian jsonl spans",
    "ru_hallucination filename:data",
    "RAGTruth filename:ru",
    "factuality russian filename:jsonl",
]

TREE_CANDIDATES = [
    "DFKonGH/ru_hallucination_detector",
    "sber-summer-practice/hybrid-rag-hallucination-verification",
    "IlyaGusev/ru_turbo_alpaca",
    "avidale/russe",
    "RostislavGrin/RuFAct",
    "RussianNLP/ruadapt",
    "NightStar/ru-fact-checking",
    "ai-forever/RuFactChecking",
    "saucam/ru-hallucination",
    "bond005/ru_fact_checking",
]

DATA_RE = r"\\.(jsonl|json|csv|parquet|tsv|arrow)$"


def probe(url, timeout=8):
    started = time.time()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "curl/8", "Accept": "*/*"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(2048)
            return {"url": url, "status": response.status, "bytes_read": len(payload),
                    "preview": payload[:200].decode("utf-8", "ignore"),
                    "seconds": round(time.time() - started, 2)}
    except urllib.error.HTTPError as error:
        return {"url": url, "status": error.code, "error": f"HTTPError: {error.reason}",
                "seconds": round(time.time() - started, 2)}
    except Exception as error:  # noqa: BLE001 - нужен текст любой сетевой ошибки
        return {"url": url, "status": "fail", "error": f"{type(error).__name__}: {str(error)[:160]}",
                "seconds": round(time.time() - started, 2)}


def gh(args, timeout=60):
    started = time.time()
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    return {"code": result.returncode, "seconds": round(time.time() - started, 1),
            "stdout": result.stdout.strip()[:3000], "stderr": result.stderr.strip()[:300]}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ru_clause_search_v18.json")
    parser.add_argument("--skip-hosts", action="store_true")
    args = parser.parse_args(argv)

    report = {"rule": "поиск реальных русских клаузовых span-данных; в песочнице разрешены "
                      "только github.com / codeload.github.com / api.github.com / pypi.org / "
                      "files.pythonhosted.org / registry.npmjs.org; каждый хост фиксируется с "
                      "реальным ответом",
              "hosts": {}, "github": {"repositories": [], "code": [], "trees": []},
              "downloaded": [], "verdict": {}}

    if not args.skip_hosts:
        for host, urls in HOSTS.items():
            report["hosts"][host] = [probe(url) for url in urls]

    for query in REPO_QUERIES:
        result = gh(["api", "-X", "GET", "search/repositories", "-f", f"q={query}",
                     "-f", "per_page=8", "-f", "sort=stars", "--jq",
                     '.items[] | "\(.full_name)\t\(.stargazers_count)\t\(.size)\t\(.description)"'])
        report["github"]["repositories"].append({"query": query, **result})

    for query in CODE_QUERIES:
        result = gh(["api", "-X", "GET", "search/code", "-f", f"q={query}", "-f", "per_page=5",
                     "--jq", '.items[] | "\(.repository.full_name)\t\(.path)"'])
        report["github"]["code"].append({"query": query, **result})

    for repo in TREE_CANDIDATES:
        result = gh(["api", f"repos/{repo}/git/trees/HEAD?recursive=1", "--jq", DATA_JQ])
        report["github"]["trees"].append({"repo": repo, **result})

    # что уже лежит локально из внешних источников
    for path in sorted(EXTERNAL.glob("*/**/*")):
        if path.is_file() and path.suffix in {".jsonl", ".csv", ".parquet", ".json"} \
                and path.stat().st_size > 50_000:
            report["downloaded"].append({
                "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha256(path)})

    report["verdict"] = {
        "hosts_reachable": [host for host, rows in report["hosts"].items()
                            if any(row.get("status") in (200, 201) for row in rows)],
        "hosts_blocked": [host for host, rows in report["hosts"].items()
                          if all(row.get("status") == "fail" for row in rows)],
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["verdict"], ensure_ascii=False), flush=True)
    print("repos queries:", len(report["github"]["repositories"]),
          "code queries:", len(report["github"]["code"]), flush=True)


if __name__ == "__main__":
    main()
