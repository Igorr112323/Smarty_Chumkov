"""Проверка протокола и плана промпта 63 до запуска обучения."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    plan = json.loads((ROOT / "artifacts/plan63.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "artifacts/protocol63.json").read_text(encoding="utf-8"))
    if not protocol["stage1"]["passed"]:
        raise SystemExit("стадия 1 не пройдена: оракул ниже порога")
    if protocol["stage1"]["oracle_token_f1"] < 0.75:
        raise SystemExit("оракул 1.4 ниже 0,75 — обучение запускать нельзя")
    if not plan.get("frozen_before_run"):
        raise SystemExit("план не зафиксирован до прогона")
    if plan["criteria"]["3.4_span_f1_soft"] != 0.35:
        raise SystemExit("критерий 3.4 не соответствует уточнению заказчика (0,35)")
    print("протокол ok, оракул", protocol["stage1"]["oracle_token_f1"])
    print("критерии:", json.dumps(plan["criteria"], ensure_ascii=False))


if __name__ == "__main__":
    main()
