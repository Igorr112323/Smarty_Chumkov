"""Стадия 2: приёмка поставки.

Что проверяется:
 1. Путь заказчика (`spanver.tiny_backend.TinyBundle`, он же CLI/HTTP) против
    независимого пути проекта (`tools/nested_cv35.predict` + `spanver.snap`).
    Разные реализации одного конвейера, одни и те же веса, одни и те же пары.
 2. Метрики считаются ТОЛЬКО на внешнем фолде 1 (140 пар), которые seed 0 не
    видел при обучении. Метрика на всех 699 парах — внутриобучающая и как
    оценка качества не приводится.

Критерий приёмки: |token F1 поставки - token F1 независимого пути| <= 0.005.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.tiny_backend import TinyBundle
from tools.decode52 import bootstrap, components
from tools.nested_cv35 import predict
from tools.train_cv_feat import token_fpr


def span_metrics(pairs, predicted):
    exact = overlap = iou = predicted_spans = gold_spans = correct = matched = 0
    rows = []
    for pair in pairs:
        answer = pair["model_output"]
        spans = predicted[pair["pair_id"]]
        gold = pair["spans"]
        predicted_spans += len(spans)
        gold_spans += len(gold)
        used = set()
        for span in spans:
            best, index = 0.0, None
            for position, truth in enumerate(gold):
                if position in used:
                    continue
                intersection = min(span["end"], truth["end"]) - max(span["start"], truth["start"]) + 1
                if intersection <= 0:
                    continue
                union = max(span["end"], truth["end"]) - min(span["start"], truth["start"]) + 1
                value = intersection / union
                if value > best:
                    best, index = value, position
            if index is None:
                continue
            used.add(index)
            overlap += 1
            if best == 1.0:
                exact += 1
            if best >= 0.5:
                iou += 1
                matched += 1
                if gold[index]["type"] == span["type"]:
                    correct += 1
        gold_tokens = covered_tokens(answer, [(s["start"], s["end"]) for s in gold])
        taken = covered_tokens(answer, [(s["start"], s["end"]) for s in spans])
        rows.append({"doc_ids": pair["doc_ids"], "support": {
            "token_tp": len(gold_tokens & taken),
            "token_fp": len(taken - gold_tokens),
            "token_fn": len(gold_tokens - taken),
            "negative_tokens": len(token_positions(answer)) - len(gold_tokens),
            "flagged_negative_tokens": len(taken - gold_tokens),
        }})
    denominator = predicted_spans + gold_spans
    tp = sum(row["support"]["token_tp"] for row in rows)
    fp = sum(row["support"]["token_fp"] for row in rows)
    fn = sum(row["support"]["token_fn"] for row in rows)
    negatives = sum(row["support"]["negative_tokens"] for row in rows)
    flagged = sum(row["support"]["flagged_negative_tokens"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    negative_pairs = sum(1 for pair in pairs if not pair["spans"])
    flagged_pairs = sum(1 for pair in pairs if not pair["spans"] and predicted[pair["pair_id"]])
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr_token": flagged / negatives if negatives else 0.0,
        "fpr_pair": flagged_pairs / negative_pairs if negative_pairs else 0.0,
        "span_exact_f1": 2 * exact / denominator if denominator else 0.0,
        "span_overlap_f1": 2 * overlap / denominator if denominator else 0.0,
        "span_iou_f1": 2 * iou / denominator if denominator else 0.0,
        "type_accuracy": correct / matched if matched else 0.0,
        "predicted_spans": predicted_spans,
        "gold_spans": gold_spans,
        "bootstrap": bootstrap(components(rows)),
    }


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-spans", type=int, default=3)
    parser.add_argument("--out", default="artifacts/cli52.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    from tools.train_base33 import grouped_folds

    folds = grouped_folds(pairs, 5)
    held = folds[0]
    head = json.loads((ROOT / "artifacts/tiny_head.json").read_text(encoding="utf-8"))
    threshold = head["start"] if args.threshold is None else args.threshold
    gate = head["gate"]

    bundle = TinyBundle(None, ROOT / "artifacts/tiny_head.json")
    from tools.predict_frozen38 import load_state

    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    if entry["seed"] != 0:
        raise ValueError("в манифесте первым должен стоять seed 0")
    load_state(torch, bundle.model, ROOT / "weights/final" / f"seed_{entry['seed']}", entry)

    delivered = {}
    for pair in held:
        payload = bundle.verify(pair["model_output"], pair["document_text"], max_spans=args.max_spans)["spans"]
        # сужение до max_spans делает сам бэкенд; snapped здесь идемпотентен
        delivered[pair["pair_id"]] = payload

    cache = {}
    for pair in held:
        cached = bundle.lexfeat.pair_probabilities(bundle.tokenizer, bundle.model, pair, top=head["top_chunks"])
        cache[pair["pair_id"]] = cached
    reference = {}
    for pair in held:
        value = cache[pair["pair_id"]]
        if value is None or not value["pieces"]:
            reference[pair["pair_id"]] = []
            continue
        if max(piece[2] for piece in value["pieces"]) < gate:
            reference[pair["pair_id"]] = []
            continue
        spans = predict(value, threshold)
        reference[pair["pair_id"]] = sorted(snapped(spans, pair["model_output"], "expand"),
                                            key=lambda item: (-item["risk"], item["start"]))[:args.max_spans]

    delivered_metrics = span_metrics(held, delivered)
    reference_metrics = span_metrics(held, reference)
    delta = delivered_metrics["token_f1"] - reference_metrics["token_f1"]
    report = {
        "status": "готово",
        "scope": "внешний фолд 1 (140 пар), которые seed 0 не видел; 699 пар не оценивались как качество",
        "pairs": len(held),
        "threshold": threshold,
        "gate": gate,
        "max_spans": args.max_spans,
        "delivered": delivered_metrics,
        "reference": reference_metrics,
        "token_f1_delta": delta,
        "acceptance_limit": 0.005,
        "accepted": abs(delta) <= 0.005 and delivered_metrics["token_precision"] == reference_metrics["token_precision"],
        "test301_evaluated": False,
        "external_data_used": False,
        "note": "token_fpr из tools.train_cv_feat не используется: FPR считается по той же маске, что и token F1, чтобы расхождение путей было видно",
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pairs", "threshold", "gate", "max_spans", "token_f1_delta", "accepted")}, ensure_ascii=False, indent=1))
    for name in ("delivered", "reference"):
        value = report[name]
        print(f"{name:10s} token F1 {value['token_f1']:.4f} P {value['token_precision']:.4f} R {value['token_recall']:.4f} "
              f"FPRt {value['fpr_token']:.4f} FPRp {value['fpr_pair']:.4f} span_exact {value['span_exact_f1']:.4f} "
              f"span_iou {value['span_iou_f1']:.4f} тип {value['type_accuracy']:.4f} "
              f"CI [{value['bootstrap']['token_f1_ci95'][0]:.4f};{value['bootstrap']['token_f1_ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
