import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from spanver import lexfeat


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/cv/feat5/fold_1")
    parser.add_argument("--out", default="artifacts/checkpoints/feat5_fold1")
    parser.add_argument("--recipe", default="python tools/train_cv_feat.py --folds 5 --epochs 1 --report artifacts/cv_report_feat5.json")
    parser.add_argument("--metrics", default="artifacts/cv_report_feat5.json")
    args = parser.parse_args(argv)

    source = Path(args.checkpoint)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    saved = torch.load(source / "feature_model.pt", map_location="cpu")
    head = {name: tensor for name, tensor in saved["state_dict"].items() if name.startswith("proj") or "classifier" in name}
    torch.save({"head": head, "source": saved.get("source")}, target / "head.pt")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        path = source / name
        if path.exists():
            (target / name).write_bytes(path.read_bytes())
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = {}
    for path in sorted(target.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8")) if Path(args.metrics).exists() else {}
    first = next((row for row in metrics.get("rows", []) if row.get("fold") == 1), {})
    manifest = {
        "checkpoint": str(target),
        "fold": 1,
        "pairs_held_out": first.get("held_pairs"),
        "folds": metrics.get("folds"),
        "epochs": metrics.get("epochs"),
        "seed": 0,
        "git_sha": git_sha,
        "git_sha_checkpoint_commit": "38e1db2",
        "recipe": args.recipe,
        "base_weights": "models/rubert-tiny2 (sha256 137fa2b1d944dae19c74456dfe8fac2f780d9acf34e037f5d1e37acba1157768), собирается из weights/rubert-tiny2: python tools/assemble_weights.py weights/rubert-tiny2 models/rubert-tiny2",
        "data": "artifacts/gold.jsonl (sha256 fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b) + artifacts/rufacts/rufacts_negative.jsonl (2994)",
        "frozen_encoder": "полные веса энкодера не хранятся в git: git-lfs в песочнице недоступен (GitHub release assets закрыты), файл 117 МБ > лимита GitHub 100 МБ на файл",
        "restore": "python tools/train_cv_feat.py --folds 5 --epochs 1 --report artifacts/cv_report_feat5.json (детерминированно, повтор дал те же метрики фолдов 1-3)",
        "head_only": True,
        "metrics_fold_1": {"token_f1": first.get("token_f1"), "span_f1": first.get("span_f1"), "fpr_pair": first.get("fpr_pair"), "fpr_token": first.get("fpr_token")},
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
