import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def decide(macro_f1, pooled_f1, pooled_fpr, remaining_minutes):
    if macro_f1 >= 0.5 and pooled_fpr <= 0.4:
        return "cv_success" if pooled_f1 >= 0.5 else "stop_metric_mismatch"
    if macro_f1 < 0.47:
        return "stop_keep_v2"
    if 0.48 <= macro_f1 < 0.5:
        return "run_v5" if remaining_minutes > 70 else "stop_budget"
    return "stop_unspecified"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-start", required=True)
    args = parser.parse_args()
    summary = json.loads((ROOT / "artifacts/ablation47_v4_summary.json").read_text(encoding="utf-8"))
    pooled = json.loads((ROOT / "artifacts/ablation47_v4_pooled.json").read_text(encoding="utf-8"))
    if summary["status"] != "готово" or len(summary["folds"]) != 5:
        raise ValueError("В4 не завершён")
    start = datetime.fromisoformat(args.budget_start.replace("Z", "+00:00"))
    remaining = 180 - (datetime.now(timezone.utc) - start).total_seconds() / 60
    macro_f1 = statistics.mean(row["metrics"]["token_f1"] for row in summary["folds"])
    pooled_f1 = pooled["pooled_token_f1_exact"]
    fpr = pooled["pooled_fpr_token"]
    result = {"macro_token_f1_from_rounded_fold_metrics": macro_f1, "pooled_token_f1": pooled_f1, "pooled_fpr_token": fpr, "budget_start": args.budget_start, "checked_at": datetime.now(timezone.utc).isoformat(), "remaining_minutes": remaining, "decision": decide(macro_f1, pooled_f1, fpr, remaining), "contract_confirmed": False}
    (ROOT / "artifacts/ablation47_decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
