import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
artifacts = ROOT / "artifacts"
summary = json.loads((artifacts / "nested38_summary.json").read_text(encoding="utf-8"))
folds = []
for number in range(1, 6):
    path = artifacts / f"reproduce45_fold{number}.json"
    folds.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
complete = all(fold and fold.get("status") == "готово" and len(fold.get("seeds", [])) == 5 for fold in folds)
verification_path = artifacts / "final45_restore_verification.json"
verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
result = {"folds_complete": sum(bool(fold and fold.get("status") == "готово" and len(fold.get("seeds", [])) == 5) for fold in folds), "folds_expected": 5, "all_metrics_match_original": complete and all(fold["metrics_equal"] for fold in folds), "restore_verified": bool(verification and verification.get("matches_original")), "fold_files": [f"artifacts/reproduce45_fold{number}.json" if fold else None for number, fold in enumerate(folds, 1)], "test301": False, "external_data": False}
if complete:
    result["macro_token_f1"] = round(statistics.mean(fold["metrics"]["token_f1"] for fold in folds), 4)
    result["expected_macro_token_f1"] = summary["summary"]["token_f1"]["mean"]
    result["max_fpr_token"] = max(fold["metrics"]["fpr_token"] for fold in folds)
    result["max_fpr_pair"] = max(fold["metrics"]["fpr_pair"] for fold in folds)
(artifacts / "final45_reproduction_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False), flush=True)
