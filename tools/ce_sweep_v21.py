"""V21: короткий sweep гиперпараметров cross-encoder на реальных парах (CPU, ~1 мин/эпоха).

Выбор — по val (наш dev-набор): AUC риска CE и Span в декадере v11 с порогами в бюджете FPR <= 0.40.
Held-out не участвует. Лучший чекпойнт сохраняется отдельно.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch
from sklearn.metrics import roc_auc_score

import sweep_v16 as S
import v11_lab as L
import ce_train_cpu_v21 as C
from ce_model_v21 import build_model, load_tokenizer


CONFIGS = (
    {"name": "base_e8", "epochs": 8, "batch": 32, "lr": 5e-5, "max_easy": 2, "doc": 64, "len": 128},
    {"name": "easy4_e8", "epochs": 8, "batch": 32, "lr": 5e-5, "max_easy": 4, "doc": 64, "len": 128},
    {"name": "long_doc96", "epochs": 6, "batch": 32, "lr": 5e-5, "max_easy": 2, "doc": 96, "len": 192},
    {"name": "lr3_e10", "epochs": 10, "batch": 32, "lr": 3e-5, "max_easy": 2, "doc": 64, "len": 128},
    {"name": "lr3_e14", "epochs": 14, "batch": 32, "lr": 3e-5, "max_easy": 2, "doc": 64, "len": 128},
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ce_sweep_v21.json")
    parser.add_argument("--checkpoint", default="artifacts/ce_model_cpu_best_v21.pt")
    parser.add_argument("--configs", default="")
    parser.add_argument("--include-synth", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(int(os.environ.get("CE_THREADS", "2")))
    tokenizer = load_tokenizer()
    train_pairs, val_pairs, _ = L.load_all()
    v19 = C.read_jsonl(ROOT / "artifacts/ru_clause_gen_v19.jsonl")
    synth = C.read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl") if args.include_synth else []
    pairs = train_pairs + (v19 if args.include_synth else []) + synth

    tables_val = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables.json")
    configs = list(CONFIGS)
    if args.configs:
        wanted = {name.strip() for name in args.configs.split(",") if name.strip()}
        configs = [config for config in configs if config["name"] in wanted]

    report = {"rule": "sweep на реальных парах (+опционально v19/synth), выбор по val AUC и Span "
                      "(пороги в бюджете FPR <= 0.40); held-out не читается",
              "train_pairs": len(pairs), "runs": {}, "best": None}
    best = None
    for config in configs:
        started = time.time()
        tensors, meta = C.pack(pairs, tokenizer, config["doc"], config["len"], config["max_easy"])
        val_tensors, val_meta = C.pack(val_pairs, tokenizer, config["doc"], config["len"], max_easy=-1)
        model = build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
        history = []
        for epoch in range(config["epochs"]):
            loss = C.run_epoch(model, tensors, optimizer, config["batch"])
            history.append({"epoch": epoch, "loss": loss})
        values = C.scores(model, val_tensors)
        labels = [entry["label"] for entry in val_meta]
        auc = float(roc_auc_score(labels, values))
        C.apply_to_tables(tables_val, val_meta, values)
        cell = C.contract(val_pairs, tables_val)
        entry = {"config": config, "examples": int(tensors["risk"].shape[0]),
                 "seconds": round(time.time() - started, 1), "val_auc": round(auc, 4),
                 "val": cell, "history": history}
        report["runs"][config["name"]] = entry
        print(config["name"], "auc", entry["val_auc"], "span", (cell or {}).get("span"),
              f"{entry['seconds']} c", flush=True)
        score = ((cell or {}).get("span") or 0, auc)
        if best is None or score > best[0]:
            best = (score, config["name"], model, config)
            torch.save({"state_dict": model.state_dict(), "config": config},
                       ROOT / args.checkpoint)
            report["best"] = {"name": config["name"], "val_auc": entry["val_auc"],
                              "val": cell, "checkpoint": args.checkpoint, "config": config}
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print("best", report["best"], flush=True)


if __name__ == "__main__":
    main()
