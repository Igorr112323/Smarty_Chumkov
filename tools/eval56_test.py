"""ПРОМПТ 56, критерий 2.8: финальный замер на test 301 в двух видах.

Правила, которые здесь соблюдаются жёстко:

  * test 301 НЕ используется ни для чего, кроме этого замера: ни порог, ни
    конфигурация, ни выбор варианта здесь не подбираются. Порог берётся из
    манифеста фолда (`weights/ablation51/ten/fold{N}/manifest.json`), то есть
    выбранного на внутренних фолдах;
  * отчёт печатает ОБА числа: все 301 пара и «чистая» подвыборка (пары, чьи
    документы не встречаются в train+val), размер подвыборки указан;
  * отдельной строкой идёт тот же замер на старом входе (joined_reference),
    чтобы сравнение было парным.

Скрипт запускается в Actions: нужен torch и восстановленные веса фолдов.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import read_jsonl, token_positions
from spanver.snap import snapped
from tools.nested_cv35 import GATE, averaged, probabilities
from tools.sweep_ragtruth30 import compute, prepare
from tools.train_base33 import grouped_folds


def build_document(docs):
    blocks = []
    for item in docs:
        ident = str(item.get("doc_id") or "").strip()
        body = item.get("doc_text") or ""
        blocks.append(f"## {ident}\n{body}" if ident else body)
    return "\n\n".join(blocks)


def parse_docs(raw):
    import ast
    text = str(raw or "").strip()
    if not text.startswith("["):
        return [{"doc_id": "", "doc_text": text}] if text else []
    value = ast.literal_eval(text)
    if isinstance(value, dict):
        value = [value]
    return [item for item in value if isinstance(item, dict)]


def load_test_rows(entry, pairs_path=None):
    """Собирает 301 тестовую пару с нужным входом.

    Метки всегда берутся из исходных CSV (они единственный источник истины).
    Вход: при entry=docs — склейка всех документов поля docs; при
    entry=joined_reference — поле paired-файла, если он передан (ours33.jsonl),
    иначе joined_reference из CSV. Подмена входа не меняет ни одной метки.
    """
    provided = {}
    if pairs_path:
        from spanver.evaluate import read_jsonl as _read
        provided = {row["pair_id"]: row for row in _read(ROOT / pairs_path)}

    import json as _json

    from spanver.answer_json import parse_answer
    from spanver.io_csv import read_gold
    from spanver.locate import locate
    from spanver.splits import load_splits
    from spanver.types import classify

    splits = load_splits()
    test_ids = set(splits["test"])
    rows = []
    files = (
        ("sberquad-rag.csv", "sberquad", "sber", "1_rushallu_rag_gold/sberquad-rag.csv"),
        ("ruscibench-rag.csv", "ruscibench", "rusci", "1_rushallu_rag_gold/ruscibench-rag.csv"),
    )
    for filename, domain, prefix, source in files:
        for number, row in enumerate(read_gold(ROOT / "Dataset" / "1_rushallu_rag_gold" / filename)):
            pair_id = f"{prefix}{number:06d}"
            if pair_id not in test_ids:
                continue
            output = row.get("model_output") or ""
            raw = parse_answer(row.get("answer"), pair_id)
            spans = []
            for item, found in zip(raw, locate(raw, output)):
                spans.append({"text": item["text"], "start": found["start"], "end": found["end"],
                              "type_src": item["type"], "type": classify(item["text"], item["type"]),
                              "evidence_ids": []})
            if entry == "docs":
                document = build_document(parse_docs(row.get("docs")))
            elif provided:
                document = provided[pair_id]["document_text"]
            else:
                document = row.get("joined_reference") or ""
            rows.append({"pair_id": pair_id, "domain": domain, "source": source,
                         "document_text": document, "model_output": output, "spans": spans})
    return rows


def clean_subset(test_rows):
    """Пары test, чьи документы не встречаются в train+val."""
    import ast

    from spanver.io_csv import read_gold
    from spanver.splits import load_splits

    splits = load_splits()
    train_val = set(splits["train"]) | set(splits["val"])
    files = (
        ("sberquad-rag.csv", "sber", "1_rushallu_rag_gold/sberquad-rag.csv"),
        ("ruscibench-rag.csv", "rusci", "1_rushallu_rag_gold/ruscibench-rag.csv"),
    )
    docs_of = {}
    for filename, prefix, source in files:
        for number, row in enumerate(read_gold(ROOT / "Dataset" / "1_rushallu_rag_gold" / filename)):
            pair_id = f"{prefix}{number:06d}"
            text = str(row.get("docs") or "").strip()
            value = ast.literal_eval(text) if text.startswith("[") else []
            docs_of[pair_id] = {str(item.get("doc_id")) for item in value if isinstance(item, dict)}
    train_docs = set().union(*[docs_of[pair_id] for pair_id in train_val if pair_id in docs_of]) \
        if train_val else set()
    clean = [row["pair_id"] for row in test_rows if not (docs_of.get(row["pair_id"], set()) & train_docs)]
    return set(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", choices=("docs", "joined_reference"), required=True)
    parser.add_argument("--pairs", default=None,
                        help="для entry=joined_reference подать artifacts/ours33.jsonl")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    from spanver import lexfeat
    from tools.predict_frozen38 import load_state

    torch.use_deterministic_algorithms(True)
    rows = load_test_rows(args.entry, args.pairs)
    clean = clean_subset(rows)
    path = Path(args.out) if args.out else ROOT / f"artifacts/eval56_test_{args.entry}.json"
    report = {"status": "в процессе", "entry": args.entry, "fold": args.folds,
              "test_pairs": len(rows),
              "clean_pairs": len(clean), "fold_metrics": [], "test301": True,
              "usage": "только финальный замер; порог из манифеста внутренних фолдов"}
    thresholds = {}

    for fold in [int(item) for item in args.folds.split(",")]:
        folder = ROOT / "weights/ablation51/ten" / f"fold{fold}"
        if not (folder / "manifest.json").exists():
            raise SystemExit(f"нет манифеста фолда {fold}")
        thresholds[fold] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        # порог выбран на внутренних фолдах тем же правилом, что в decode56
        threshold = thresholds[fold].get("threshold") or {}
        if not isinstance(threshold, dict) or "start" not in threshold:
            raise SystemExit(f"в манифесте фолда {fold} нет выбранного порога")
    # тестовые пары не входят ни в один train-фолд, поэтому каждый фолд-детектор
    # применим ко всем 301; берём ансамбль по сидам каждого фолда и усредняем
    # вероятности по всем фолдам так же, как в отборе порога
    caches = []
    for fold in sorted(thresholds):
        manifest = thresholds[fold]
        seed_caches = []
        for entry in manifest["checkpoints"]:
            tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=entry["seed"])
            model.base.config._attn_implementation = "eager"
            load_state(torch, model, ROOT / "weights/ablation51/ten" / f"fold{fold}" / f"seed_{entry['seed']}", entry)
            seed_caches.append(probabilities(tokenizer, model, rows))
            del model, tokenizer
        caches.append(seed_caches)

    # кэши считаются по всем 301 паре один раз; подвыборка фильтруется после
    fold_caches = {}
    for fold in sorted(thresholds):
        fold_caches[fold] = {row["pair_id"]: averaged(row, caches[fold - 1], range(len(caches[fold - 1])))
                             for row in rows}

    def measure(subset):
        """Каждый фолд-детектор применяется ко всем парам подвыборки, затем счётчики
        объединяются (pooled), а не усредняются: усреднение F1 по фолдам завышает
        результат при разном объёме поддержки."""
        result = {}
        totals = {"token_tp": 0, "token_fp": 0, "token_fn": 0,
                  "negative_tokens": 0, "flagged_negative_tokens": 0,
                  "span_tp": 0, "span_fp": 0, "span_fn": 0,
                  "negative_pairs": 0, "flagged_negative_pairs": 0}
        for fold in sorted(thresholds):
            manifest = thresholds[fold]
            prepared = prepare(subset, fold_caches[fold])
            point = compute(prepared, manifest["threshold"]["start"],
                            manifest["threshold"]["extend"], manifest["gate"])
            result[str(fold)] = point
            for key in totals:
                totals[key] += point["support"].get(key, 0)
        tp, fp, fn = totals["token_tp"], totals["token_fp"], totals["token_fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        span_denom = 2 * totals["span_tp"] + totals["span_fp"] + totals["span_fn"]
        pooled = {
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr_token": totals["flagged_negative_tokens"] / totals["negative_tokens"]
            if totals["negative_tokens"] else None,
            "fpr_pair": totals["flagged_negative_pairs"] / totals["negative_pairs"]
            if totals["negative_pairs"] else None,
            "span_exact_f1": 2 * totals["span_tp"] / span_denom if span_denom else 0.0,
            "counts": totals,
            "note": "pair-метрика здесь не считается: tagged-счётчики support не дают пар без спанов",
        }
        per_fold = {name: statistics.pstdev([result[key][name] for key in result
                                             if result[key].get(name) is not None])
                    for name in ("token_f1_exact", "fpr_token")} if len(result) > 1 else {}
        return {"folds": result, "pooled": pooled, "std_over_folds": per_fold,
                "pairs": len(subset)}

    report["all_pairs"] = measure(rows)
    report["clean_subset"] = measure([row for row in rows if row["pair_id"] in clean])
    report["status"] = "готово"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"entry": args.entry, "test_pairs": len(rows), "clean_pairs": len(clean)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
