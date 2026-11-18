"""ПРОМПТ 56, критерий 2.7: скорость на новом входе.

Норма п.7 двусторонняя и проверяется здесь целиком:

  * мс на 1000 знаков документа <= 30;
  * прогон 2 000 000 знаков <= 60 с.

Отличие от `speed52.py`: там норма мерилась на ОДНОМ сиде (`--seeds 1,3,5` —
это три прогона с разным числом моделей, лимит применялся к строке models=1),
длин было четыре и не было сценария «2 млн знаков за один прогон». Здесь
длины берутся из фактического распределения нового входа (838, 1000, 3980,
5000, 20000), а 2 млн знаков — отдельный сценарий: реальные пары повторяются
до 2 000 000 знаков суммарно, измеряется всё время прогона.

Чтобы сравнение с 52 было парным, скорость считается для 1, 3 и 5 сидов
сразу: строка `models=1` сопоставима с прежней нормой напрямую, строки 3 и 5
показывают цену ансамбля, которым получены метрики 2.1–2.6. Приёмка 2.7
ставится по одному сиду — так же, как норма понималась в 52.

Веса берутся из `weights/ablation51/ten/fold{N}`; порог — из манифеста фолда,
то есть выбранный на внутренних фолдах, а не на test.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from tools.nested_cv35 import predict
from tools.speed_bench_batched import build_pairs

CHUNKS_TOP = 4
LIMIT_MS_PER_1000 = 30.0
LIMIT_2M_SECONDS = 60.0
TWO_MILLION = 2_000_000
COUNTS = (1, 3, 5)


def load_fold(fold, count=5):
    import torch

    from tools.predict_frozen38 import load_state

    folder = ROOT / "weights/ablation51/ten" / f"fold{fold}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["checkpoints"][:count]
    if len(entries) != count:
        raise SystemExit(f"фолд {fold}: в манифесте {len(manifest['checkpoints'])} сидов, нужно {count}")
    models = []
    for entry in entries:
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=entry["seed"])
        model.base.config._attn_implementation = "eager"
        load_state(torch, model, folder / f"seed_{entry['seed']}", entry)
        models.append((entry["seed"], tokenizer, model))
    return models, manifest


def merge(caches, pair):
    pieces = [cache[pair["pair_id"]]["pieces"] for cache in caches]
    return [[piece[0], piece[1], sum(item[position][2] for item in pieces) / len(pieces)]
            for position, piece in enumerate(pieces[0])]


def run_pass(models, pairs, theta, gate):
    """Один полный прогон: вероятности по всем сидам плюс решающее правило."""
    caches = [lexfeat.batched_probabilities(tokenizer, model, pairs, top=CHUNKS_TOP, batch=16)
              for _, tokenizer, model in models]
    missing = [pair["pair_id"] for pair in pairs if any(pair["pair_id"] not in cache for cache in caches)]
    if missing:
        raise SystemExit(f"нет вероятностей для {len(missing)} пар, первая {missing[0]}")
    for pair in pairs:
        pieces = merge(caches, pair)
        answer = pair["model_output"]
        spans = []
        if max((piece[2] for piece in pieces), default=0.0) >= gate:
            spans = predict({"answer": answer, "pieces": pieces}, theta)
        snapped(spans, answer, "expand")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--lengths", default="838,1000,3980,5000,20000")
    parser.add_argument("--counts", default="1,3,5")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-two-million", action="store_true")
    parser.add_argument("--out", default="artifacts/speed56.json")
    args = parser.parse_args()
    print(f"speed56: вход {args.pairs}, фолд {args.fold}, сидов {args.counts}", flush=True)

    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    gold = read_jsonl(ROOT / args.pairs)
    counts = [int(item) for item in args.counts.split(",")]
    models, manifest = load_fold(args.fold, max(counts))
    theta = manifest["threshold"]["start"]
    gate = manifest["gate"]
    lengths = [int(item) for item in args.lengths.split(",")]
    print(f"порог {theta}, гейт {gate}, сидов загружено {len(models)}", flush=True)

    rows = []
    for length in lengths:
        pairs = build_pairs(gold, length)
        chars = sum(len(pair["document_text"]) for pair in pairs)
        for count in counts:
            repeats = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                run_pass(models[:count], pairs, theta, gate)
                repeats.append((time.perf_counter() - started) * 1000.0 / (chars / 1000.0))
            rows.append({
                "models": count,
                "length": length,
                "documents": len(pairs),
                "doc_chars": chars,
                "answer_chars": sum(len(pair["model_output"]) for pair in pairs),
                "ms_per_1000_doc_median": statistics.median(repeats),
                "ms_per_1000_doc_min": min(repeats),
                "ms_per_1000_doc_max": max(repeats),
                "samples": repeats,
                "passes_limit": statistics.median(repeats) <= LIMIT_MS_PER_1000,
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    two_million = []
    if not args.skip_two_million:
        # в сценарий идут только пары с непустым ответом: без ответа
        # batched_probabilities не выдаёт записей и ключа в кэше не появляется
        usable = [pair for pair in gold if pair.get("model_output") and pair.get("document_text")]
        if not usable:
            raise SystemExit("во входе нет пар с непустым ответом")
        for count in counts:
            runs = []
            for _ in range(args.repeats):
                pairs, chars, index = [], 0, 0
                while chars < TWO_MILLION:
                    base = usable[index % len(usable)]
                    pairs.append({"pair_id": str(index),
                                  "model_output": base["model_output"],
                                  "document_text": base["document_text"]})
                    chars += len(base["document_text"])
                    index += 1
                started = time.perf_counter()
                run_pass(models[:count], pairs, theta, gate)
                runs.append({"documents": len(pairs), "doc_chars": chars,
                             "seconds": time.perf_counter() - started})
                print(json.dumps({"models": count, **runs[-1]}, ensure_ascii=False), flush=True)
            two_million.append({"models": count,
                                "runs": runs,
                                "median_seconds": statistics.median(item["seconds"] for item in runs),
                                "doc_chars": runs[0]["doc_chars"],
                                "documents": runs[0]["documents"],
                                "passes_limit": statistics.median(item["seconds"] for item in runs)
                                <= LIMIT_2M_SECONDS})

    single = [row for row in rows if row["models"] == 1]
    single_two = [item for item in two_million if item["models"] == 1]
    report = {
        "status": "готово",
        "entry": "docs",
        "fold": args.fold,
        "models": len(models),
        "top_chunks": CHUNKS_TOP,
        "theta": theta,
        "gate": gate,
        "threads": 4,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "limit_ms_per_1000": LIMIT_MS_PER_1000,
        "limit_2m_seconds": LIMIT_2M_SECONDS,
        "rows": rows,
        "two_million": two_million,
        # приёмка по одному сиду: так же, как норма понималась в speed52
        "acceptance": {
            "2.7_ms_per_1000_le_30": bool(single) and all(row["passes_limit"] for row in single),
            "2.7_two_million_le_60s": bool(single_two) and single_two[0]["passes_limit"],
        },
        "note": ("строка models=1 сопоставима с прежней нормой 52 (там лимит применялся "
                 "к одному сиду); models=3 и models=5 показывают цену ансамбля, "
                 "которым получены метрики 2.1–2.6"),
        "test301": False,
    }
    report["acceptance_all"] = all(report["acceptance"].values())
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report["acceptance"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
