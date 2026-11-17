import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("control", "v1", "v2", "v3"))
    parser.add_argument("--run", required=True, type=int)
    args = parser.parse_args()
    raw = subprocess.check_output(["gh", "run", "view", str(args.run), "--json", "databaseId,status,conclusion,startedAt,updatedAt,headSha,url"], cwd=ROOT, text=True)
    run = json.loads(raw)
    if run["status"] != "completed":
        raise ValueError("запуск ещё не завершён")
    start = datetime.fromisoformat(run["startedAt"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(run["updatedAt"].replace("Z", "+00:00"))
    run["wall_seconds"] = round((end - start).total_seconds())
    target = ROOT / "artifacts/ablation46_run_info.json"
    history = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    history[args.variant] = run
    target.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(history[args.variant], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
