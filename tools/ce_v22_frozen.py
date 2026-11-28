"""V22: прогоны с регуляризацией на rubert-base-cased (изоляция причины переобучения).

Полный прогон v22 показал: при 549 реальных парах 178M-параметровый энкодер переобучается —
train loss 1.20 → 0.14 за 6 эпох, val AUC 0.678 → 0.597, val Span 0.3411 (хуже, чем tiny2 0.3818
в v21). Этот скрипт отвечает на вопрос «дело в ёмкости или в объёме разметки» двумя прогонами:

  arm1 — энкодер заморожен целиком (обучаются только головы CE): классический линейный пробник,
         lr 1e-3, 3 эпохи;
  arm2 — заморожены нижние 8 из 12 слоёв, lr 3e-5, 3 эпохи.

Затем `decode` применяет оба чекпойнта в декодере v11 теми же вариантами, что в v21
(ce_only / blend / CE как пара-гейт), чтобы сравнение с tiny2-результатом было прямым.

Режимы: weights | arm1 | arm2 | decode.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/rubert-base-cased"
AUX = ROOT / "weights/rubert-base-cased/curadoria"
AUX_FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")


def weights():
    import hashlib

    from huggingface_hub import hf_hub_download

    MODEL.mkdir(parents=True, exist_ok=True)
    for name in AUX_FILES:
        shutil.copyfile(AUX / name, MODEL / name)
    cached = Path(hf_hub_download(repo_id="DeepPavlov/rubert-base-cased", filename="pytorch_model.bin"))
    digest = hashlib.sha256()
    with cached.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    shutil.copyfile(cached, MODEL / "pytorch_model.bin")
    manifest = {"repository": "DeepPavlov/rubert-base-cased", "size": cached.stat().st_size,
                "sha256": digest.hexdigest(), "channel": "GitHub Actions runner"}
    (MODEL / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(manifest, flush=True)


def run(arguments):
    command = [sys.executable, *arguments]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def arm1():
    run(["tools/ce_train_cpu_v22.py", "--encoder", str(MODEL), "--arms", "real",
         "--epochs", "3", "--batch", "16", "--lr", "1e-3", "--doc-tokens", "96", "--max-len", "192",
         "--max-easy", "2", "--freeze-encoder", "--eval-each-epoch",
         "--oof", "3", "--oof-epochs", "1",
         "--report", "artifacts/ce_train_v22_frozen.json",
         "--checkpoint", "artifacts/ce_model_v22_frozen.pt"])


def arm2():
    run(["tools/ce_train_cpu_v22.py", "--encoder", str(MODEL), "--arms", "real",
         "--epochs", "3", "--batch", "16", "--lr", "3e-5", "--doc-tokens", "96", "--max-len", "192",
         "--max-easy", "2", "--freeze-layers", "8", "--eval-each-epoch",
         "--oof", "3", "--oof-epochs", "1",
         "--report", "artifacts/ce_train_v22_frozen8.json",
         "--checkpoint", "artifacts/ce_model_v22_frozen8.pt"])


def decode():
    for checkpoint, report in (("artifacts/ce_model_v22_frozen.pt", "artifacts/ce_decode_v22_frozen.json"),
                               ("artifacts/ce_model_v22_frozen8.pt", "artifacts/ce_decode_v22_frozen8.json")):
        if not (ROOT / checkpoint).is_file():
            print(f"нет чекпойнта {checkpoint} — пропуск", flush=True)
            continue
        run(["tools/ce_decode_v21.py", "--encoder", str(MODEL), "--checkpoint", checkpoint,
             "--doc-tokens", "96", "--max-len", "192", "--report", report])


MODES = {"weights": weights, "arm1": arm1, "arm2": arm2, "decode": decode}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in MODES:
        raise SystemExit(f"режимы: {', '.join(MODES)}")
    MODES[argv[0]]()


if __name__ == "__main__":
    main()
