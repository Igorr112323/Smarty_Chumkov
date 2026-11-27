"""V11 блок 4: честный 3-fold OOF по документам на объединённом train.

Пул: RusHallu train (549) + mFAVA train (148) + synth v11 (3.5k). Fold
назначается по SHA-256 документа, поэтому синтетика не утекает к своему
родителю. Модель — stacker-ранкер (lambdarank) на скорах двух голов.
Оцениваются: OOF на held-out фолде и val (150), которая не участвует в обучении.
Held-out не читается.
"""
import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import lightgbm as lgb
import numpy as np

import v11_lab as L
import v11_ranker as R
import train_stacker_v11 as ST
from spanver.evaluate import iou, metrics, read_jsonl


def fold_of(pair):
    return int(hashlib.sha256(L.doc_key(pair).encode()).hexdigest(), 16) % 3


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/listwise_whole_model_v11.json")
    parser.add_argument("--cache", default="artifacts/router2_cache_v11.pkl")
    parser.add_argument("--report", default="artifacts/cv3_oof_v11.json")
    args = parser.parse_args(argv)

    params = L.weights()
    payload = json.loads((ROOT / args.model).read_text(encoding="utf-8"))
    cols = payload["columns"]
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    val = [p for p in gold if p.get("split") == "val"]
    train_pos = [p for p in gold if p.get("split") == "train" and p.get("spans")]
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    synth = read_jsonl(str(ROOT / "artifacts/rushallu_synth_v11.jsonl"))

    with open(ROOT / args.cache, "rb") as handle:
        cache = pickle.load(handle)
    missing = [p for p in synth + train_pos + mfava if p["pair_id"] not in cache]
    if missing:
        from router_v2_v11 import build_cache
        print("cache miss", len(missing), flush=True)
        cache.update(build_cache(missing, params, payload, cols))
        with open(ROOT / args.cache, "wb") as handle:
            pickle.dump(cache, handle)

    pool = train_pos + mfava + synth
    folds = {0: [], 1: [], 2: []}
    for pair in pool:
        folds[fold_of(pair)].append(pair)

    report = {"protocol": "3-fold OOF by document SHA; stacker lambdarank; val не в обучении",
              "pool": {"gold_train": len(train_pos), "mfava": len(mfava), "synth": len(synth)},
              "folds": {}, "oof_predictions": {}}
    oof_pred = {}
    for held in range(3):
        training = [pair for key in folds if key != held for pair in folds[key]]
        random.Random(11 + held).shuffle(training)
        X, y, group, meta, weights = ST.build_dataset(training, cache, params, set(),
                                                      whole_weight=3.0)
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=250, learning_rate=0.05,
                               num_leaves=15, min_child_samples=10, subsample=0.9,
                               colsample_bytree=0.9, random_state=11, label_gain=[0, 1],
                               verbose=-1, n_jobs=2)
        model.fit(X, y, group=group, sample_weight=weights)
        exported = R.export_lightgbm(model.booster_)
        for pair in folds[held] + val:
            entry = cache.get(pair["pair_id"])
            if not entry:
                continue
            for row in entry["rows"]:
                if "vector" not in row:
                    row["vector"] = np.array(ST.row_vector(entry, row, params), dtype=np.float32)
                row["stack"] = R.predict_raw(exported, list(row["vector"]))
        config = {"threshold": 0.5, "max_spans": 3, "gate": 0.6}
        fold_metric = metrics(folds[held], lambda pair: ST.decode(cache[pair["pair_id"]], config))
        val_metric = metrics(val, lambda pair: ST.decode(cache[pair["pair_id"]], config))
        report["folds"][str(held)] = {
            "train_pairs": len(training), "eval_pairs": len(folds[held]),
            "train_rows": int(len(y)),
            "oof": L.cell(fold_metric), "val": L.cell(val_metric),
            "model_sha256": hashlib.sha256(json.dumps(exported, sort_keys=True).encode()).hexdigest(),
        }
        for pair in folds[held]:
            oof_pred[pair["pair_id"]] = ST.decode(cache[pair["pair_id"]], config)
        print("fold", held, report["folds"][str(held)]["oof"],
              "val", report["folds"][str(held)]["val"], flush=True)

    def predictor(pair):
        return oof_pred.get(pair["pair_id"], [])

    combined = metrics(pool, predictor)
    synth_ids = {pair["pair_id"] for pair in synth}
    real = [pair for pair in pool if pair["pair_id"] not in synth_ids]
    real_metric = metrics(real, predictor)
    synth_metric = metrics([pair for pair in pool if pair["pair_id"] in synth_ids], predictor)
    by_source = {
        "real_gold_train": L.cell(metrics([p for p in real if p["pair_id"] in {x["pair_id"] for x in train_pos}], predictor)),
        "real_mfava": L.cell(metrics([p for p in real if p["pair_id"] in {x["pair_id"] for x in mfava}], predictor)),
        "real_all": L.cell(real_metric),
        "synth": L.cell(synth_metric),
        "all": L.cell(combined),
    }
    report["by_source"] = by_source
    val_cells = [report["folds"][str(key)]["val"] for key in range(3)]
    report["oof"] = L.cell(combined)
    report["val_mean"] = {name: round(sum(cell[name] for cell in val_cells) / len(val_cells), 4)
                          for name in ("span", "token", "pair", "fpr")}
    report["guard"] = {
        "oof_all": L.guard([report["oof"]]),
        "oof_real": L.guard([by_source["real_all"]]),
        "val_mean": L.guard([report["val_mean"]]),
        "both_real_and_val": L.guard([by_source["real_all"], report["val_mean"]]),
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("oof", report["oof"], flush=True)
    print("val_mean", report["val_mean"], flush=True)
    print("by_source", json.dumps(by_source, ensure_ascii=False), flush=True)
    print("guard", report["guard"], flush=True)


if __name__ == "__main__":
    main()
