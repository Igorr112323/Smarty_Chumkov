"""V17: OOF по документам для ранжировщика hgb_61 (реальные русские пары)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
import train_rank_v17 as T
import sweep_v16 as S
import v11_lab as L
from v11_lab import doc_key

L.set_weights(L.weights())
train, val, mfava = L.load_all()
sber = T.read_jsonl(ROOT / "artifacts/sber_flip_v16.jsonl")[:6000]
groups = {}
for name, pairs, cache in (("sber", sber, T.CACHE17 / "sber37_aligned.npz"),
                           ("real", train, T.CACHE14 / "real.npz"),
                           ("synth", T.read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"), T.CACHE14 / "synth.npz"),
                           ("pseudo", T.read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"), T.CACHE14 / "pseudo.npz"),
                           ("val", val, T.CACHE14 / "val.npz"),
                           ("mfava", mfava, T.CACHE14 / "mfava.npz")):
    groups[name] = T.build_group(name, pairs, cache, with_v15=(name != "sber"))

report = json.loads((ROOT / "artifacts/ranker_train_v17.json").read_text())
best_cfg = report["variants"]["hgb_61"]["config"]
docs = sorted({doc_key(p) for p in train})
np.random.RandomState(17).shuffle(docs)
fold_of = {d: i % 3 for i, d in enumerate(docs)}
real = groups["real"]
by_id = {p["pair_id"]: p for p in real["pairs"]}
fold_rows = []
folds=[]
for entry in real["meta"]:
    folds.extend([fold_of[doc_key(by_id[entry["pair_id"]])]]*entry["n"])
folds = np.asarray(folds)
X61 = np.hstack([real["X37"], real["X17"], real["X15"]])
Xs, ys = T.stack(groups, ["sber", "synth", "pseudo"], "61")
rows=[]
for fold in range(3):
    X_train = np.vstack([Xs, X61[folds != fold]]); y_train = np.concatenate([ys, real["y"][folds != fold]])
    model = T.fit_hgb(X_train, y_train)
    scores = model.predict_proba(X61[folds == fold])[:, 1]
    held_ids = {p["pair_id"] for p in train if fold_of[doc_key(p)] == fold}
    held_pairs = [p for p in train if p["pair_id"] in held_ids]
    tables = [t for t in S.build_tables(train, L.weights(), T.CACHE16 / "real_tables.json") if t["pair_id"] in held_ids]
    index = 0
    by = {t["pair_id"]: t for t in tables}
    for entry in real["meta"]:
        if entry["pair_id"] not in held_ids:
            continue
        table = by[entry["pair_id"]]
        for row in table["rows"]:
            row["risk_new"] = float(scores[index]); index += 1
    cell = L.cell(S.evaluate(held_pairs, tables, best_cfg, "risk_new"))
    cell.update({"fold": fold, "pairs": len(held_pairs)})
    rows.append(cell)
    print("fold", fold, {k: round(cell[k], 4) for k in ("span","token","pair","fpr")}, flush=True)
mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("span","token","pair","fpr")}
out = {"rule": "3 фолда по документам, hgb_61 (37+14+10), конфиг из val",
       "folds": rows, "mean": mean, "baseline_oof_span_v14": 0.2437,
       "pass": bool(mean["span"] >= 0.2437 + 0.02)}
(ROOT / "artifacts/oof_ranker_v17.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
print("mean", mean, "pass", out["pass"])
