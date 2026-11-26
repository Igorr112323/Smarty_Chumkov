"""Аудит потолка лексического ранкера: recall -> rank -> gate -> decode.

Только train/val/mFAVA-train; held-out не читается. Главный вопрос: если бы
ранкер идеально выбирал среди существующих кандидатов, какой был бы Span F1?
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans

W = json.loads((ROOT / "spanver/ranker_weights.json").read_text(encoding="utf-8"))
ranker._WEIGHTS = W


def baseline_candidates(pair):
    answer = pair.get("model_output") or ""
    return list(candidate_spans(answer))


def v4_candidates(pair):
    answer = pair.get("model_output") or ""
    return list(candidates_v4(answer))


def evaluate(pairs, candidates_fn):
    recall = collections.Counter()
    ranks = []
    oracle_spans = {}
    for pair in pairs:
        candidates = candidates_fn(pair)
        scores = [
            ranker._logit(
                ranker.features(
                    text, a, b, pair["model_output"] or "", pair["document_text"] or "",
                    *pair_context(pair),
                ), W["cand"], ranker.FEATURE_COUNT,
            )
            for a, b, text in candidates
        ]
        gold = pair.get("spans") or []
        chosen = []
        for span in gold:
            overlaps = []
            for index, (a, b, _text) in enumerate(candidates):
                value = iou(a, b, span["start"], span["end"])
                if value >= 0.5:
                    overlaps.append((value, index))
            if overlaps:
                recall["ok"] += 1
                best = max(overlaps)
                ranks.append(1 + sum(score >= scores[best[1]] for score in scores))
                chosen.append(candidates[best[1]])
            else:
                recall["miss"] += 1
        oracle_spans[pair["pair_id"]] = [
            {"start": a, "end": b, "risk": 1.0,
             "type": "unsupported_addition", "confidence": 1.0}
            for a, b, _ in chosen
        ]

    def oracle_predictor(pair):
        return oracle_spans[pair["pair_id"]]

    return {
        "oracle_recall": recall["ok"] / max(1, sum(recall.values())),
        "gold_candidates": sum(recall.values()),
        "median_gold_rank": sorted(ranks)[len(ranks) // 2] if ranks else 0,
        "top1": sum(r == 1 for r in ranks),
        "top5": sum(r <= 5 for r in ranks),
        "oracle_metrics": metrics(pairs, oracle_predictor),
        "actual_metrics": metrics(pairs, lambda p: ranker.select_spans(
            p.get("model_output") or "", p.get("document_text") or "")),
    }


CONTEXT = {}


def pair_context(pair):
    key = id(pair)
    if key not in CONTEXT:
        from spanver.module import document_context
        context = document_context(pair.get("document_text") or "")
        CONTEXT[key] = (context[0], context[2])
    return CONTEXT[key]


def compress(item):
    return {
        "oracle_recall": item["oracle_recall"],
        "gold_candidates": item["gold_candidates"],
        "median_gold_rank": item["median_gold_rank"],
        "top1": item["top1"],
        "top5": item["top5"],
        "oracle_span": item["oracle_metrics"]["span_level"]["f1"],
        "oracle_token": item["oracle_metrics"]["token_level"]["f1"],
        "oracle_pair": item["oracle_metrics"]["pair_level"]["f1"],
        "oracle_fpr": item["oracle_metrics"]["fpr"],
        "actual_span": item["actual_metrics"]["span_level"]["f1"],
        "actual_token": item["actual_metrics"]["token_level"]["f1"],
        "actual_pair": item["actual_metrics"]["pair_level"]["f1"],
        "actual_fpr": item["actual_metrics"]["fpr"],
    }


def main():
    splits = {
        "train": load_split(str(ROOT / "artifacts/gold.jsonl"), "train"),
        "val": load_split(str(ROOT / "artifacts/gold.jsonl"), "val"),
        "mfava": read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl")),
    }
    report = {"rule": "/oracle на train/val/mFAVA; held-out не используется", "splits": {}}
    for name, pairs in splits.items():
        report["splits"][name] = {
            "pairs": len(pairs),
            "baseline_candidates": compress(evaluate(pairs, baseline_candidates)),
            "v4_candidates": compress(evaluate(pairs, v4_candidates)),
        }
    target = ROOT / "artifacts/audit_ceiling_v9.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
