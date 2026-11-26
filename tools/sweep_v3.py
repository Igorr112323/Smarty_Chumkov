"""CV/val sweep для гармонизированных меток v3 (только train/val)."""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import load_split, metrics

ranker._WEIGHTS = json.loads((ROOT / "artifacts/ranker_weights_v3.json").read_text(encoding="utf-8"))
val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
sweep = []
for gate, threshold, max_spans, nms in itertools.product(
    (0.3, 0.4, 0.5, 0.6), (0.2, 0.3, 0.4, 0.5, 0.6, 0.7), (1, 2, 3, 5), (0.1, 0.3)
):
    def predictor(pair):
        return ranker.select_spans(
            pair.get("model_output") or "", pair.get("document_text") or "",
            gate=gate, threshold=threshold, max_spans=max_spans, nms_iou=nms,
        )
    result = metrics(val, predictor)
    sweep.append({
        "gate": gate, "threshold": threshold, "max_spans": max_spans, "nms_iou": nms,
        "metrics": result,
    })
sweep.sort(key=lambda row: (
    row["metrics"]["span_level"]["f1"],
    row["metrics"]["token_level"]["f1"],
), reverse=True)
report = {
    "rule": "val_v3 only; held v3 не оценивался",
    "best": sweep[:20],
    "configs": sweep,
}
target = ROOT / "artifacts/sweep_v3.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("best", report["best"][0]["metrics"]["span_level"]["f1"],
      report["best"][0]["metrics"]["token_level"]["f1"],
      report["best"][0]["metrics"]["pair_level"]["f1"],
      report["best"][0]["metrics"]["fpr"])
print("report", target, target.stat().st_size)
