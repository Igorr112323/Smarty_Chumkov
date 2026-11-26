"""Проверка порога mode selector v5 на val+mFAVA-train.

Gate и пороги global/local не меняются; variates только mode_cut.
Это отличие от предыдущих итераций: не новый ranker, а калибровка выбора
уже обученных ветвей. Held-out не используется.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import load_split, metrics, read_jsonl
from spanver.local_ranker import logistic, predict_trees
from spanver.module import candidate_spans, document_context

W = ranker._model()


def mode_probability(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    dtok, dstems, dgrams = document_context(document)
    scored = []
    for start, end, text in candidate_spans(answer):
        x = ranker.features(text, start, end, answer, document, dtok, dgrams)
        scored.append({"start": start, "end": end, "risk": ranker._logit(x[:37], W["cand"], 37)})
    return logistic(ranker.mode_features(scored, answer, document), W["mode"], 11)


def main():
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    byid = {p["pair_id"]: p for p in val + ex}
    probs = {p["pair_id"]: mode_probability(p) for p in val + ex}
    pred_global = {p["pair_id"]: ranker.select_spans(
        p.get("model_output") or "", p.get("document_text") or "",
        gate=0.5, threshold=0.6, max_spans=1, nms_iou=0.1,
    ) for p in val + ex}
    local_cfg = W["local_config"]
    pred_local = {}
    for pair in val + ex:
        pid = pair["pair_id"]
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        dtok, dstems, dgrams = document_context(document)
        items = []
        for start, end, text in candidate_spans(answer):
            if start == 0 and end == len(answer) - 1:
                continue
            x = ranker.features(text, start, end, answer, document, dtok, dgrams)
            items.append({"start": start, "end": end,
                          "risk": predict_trees(W["local_hgb"], x)})
        out = []
        for item in sorted(items, key=lambda z: -z["risk"]):
            if item["risk"] < local_cfg["threshold"]:
                break
            if any(ranker.iou(item["start"], item["end"], o["start"], o["end"]) > local_cfg["nms_iou"] for o in out):
                continue
            out.append({
                "start": item["start"], "end": item["end"],
                "risk": item["risk"], "type": "unsupported_addition", "confidence": 1.0,
            })
            if len(out) >= local_cfg["max_spans"]:
                break
        pred_local[pid] = out
    sweep = {}
    for cut in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01):
        def predict(pair):
            return pred_local[pair["pair_id"]] if probs[pair["pair_id"]] >= cut else pred_global[pair["pair_id"]]
        sweep[str(cut)] = {
            "val": metrics(val, predict),
            "mfava": metrics(ex, predict),
            "local_share_val": round(sum(probs[p["pair_id"]] >= cut for p in val) / len(val), 4),
            "local_share_mfava": round(sum(probs[p["pair_id"]] >= cut for p in ex) / len(ex), 4),
        }
    report = {
        "rule": "mode_cut only; global/local heads frozen; val+mFAVA-train selection",
        "current_mode_cut": W.get("mode_cut"),
        "sweep": sweep,
    }
    best = sorted(sweep.items(), key=lambda kv: (
        (kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"]),
        (kv[1]["val"]["token_level"]["f1"] * 6 + kv[1]["mfava"]["token_level"]["f1"]),
    ), reverse=True)
    report["best"] = [{"cut": k, **v} for k, v in best[:5]]
    for cut, item in best[:6]:
        print(cut, "val", item["val"]["span_level"]["f1"], item["val"]["token_level"]["f1"],
              "ex", item["mfava"]["span_level"]["f1"], "local shares",
              item["local_share_val"], item["local_share_mfava"])
    target = ROOT / "artifacts/mode_cut_v5.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
