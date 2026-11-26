"""Политики v5 с замороженным pair gate: добавить/заменить спаны внутри флага.

Pair gate не переобучается, поэтому FPR и pair F1 остаются как у baseline.
Пороги выбираются только на val+mFAVA-train.
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl

W = ranker._model()


def main():
    data = pickle.load(open(ROOT / "artifacts/conditional_scores_v5.pkl", "rb"))
    flags, scores = data["flags"], data["scores"]
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    pairs = train + val + ex
    baseline = {}
    for p in pairs:
        baseline[p["pair_id"]] = ranker.select_spans(p.get("model_output") or "", p.get("document_text") or "")

    def candidates(pid, local_only, max_len=None):
        answer_len = len(next(p for p in pairs if p["pair_id"] == pid).get("model_output") or "")
        items = scores.get(pid, [])
        mode = items[0][3] if items else 0.0
        local_mode = mode >= W.get("mode_cut", 0.6)
        out = []
        for s, e, score, _ in items:
            if local_mode and s == 0 and e == answer_len - 1:
                continue
            if local_only and not local_mode:
                continue
            if max_len and e - s + 1 > max_len:
                continue
            out.append((s, e, score))
        return sorted(out, key=lambda x: -x[2])

    def policy(pair, kind, th, number, max_len):
        pid = pair["pair_id"]
        if not flags.get(pid):
            return []
        out = [dict(s) for s in baseline[pid]]
        if kind == "keep":
            return out
        cs = candidates(pid, kind == "add_local", max_len)
        chosen = []
        for s, e, score in cs:
            if score < th:
                break
            if any(iou(s, e, o["start"], o["end"]) > 0.1 for o in out + chosen):
                continue
            chosen.append({
                "start": s, "end": e, "risk": score,
                "type": "unsupported_addition", "confidence": 1.0,
            })
            if len(chosen) >= number:
                break
        if kind in ("add", "add_local"):
            return out + chosen
        return chosen

    sweep = {}
    for kind in ("add", "add_local", "replace_all", "replace_local"):
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            for number in (1, 2, 3, 5):
                for max_len in (None, 40):
                    def predict(pair):
                        return policy(pair, kind, th, number, max_len)
                    mv = metrics(val, predict)
                    me = metrics(ex, predict)
                    key = f"{kind}_th{th}_n{number}_len{max_len or 'all'}"
                    sweep[key] = {"val": mv, "mfava": me}
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"],
        kv[1]["val"]["token_level"]["f1"] * 6 + kv[1]["mfava"]["token_level"]["f1"],
    ), reverse=True)
    report = {
        "rule": "pair gate frozen; val+mFAVA-train only",
        "baseline": {
            "val": metrics(val, lambda p: baseline[p["pair_id"]]),
            "mfava": metrics(ex, lambda p: baseline[p["pair_id"]]),
        },
        "best": [{"config": k, **v} for k, v in best[:10]],
        "all": sweep,
    }
    print("baseline val/mfava span", report["baseline"]["val"]["span_level"]["f1"], report["baseline"]["mfava"]["span_level"]["f1"])
    for key, item in best[:5]:
        print(key, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"], item["val"]["fpr"])
    target = ROOT / "artifacts/conditional_add_v5.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
