"""V22: обучение cross-encoder на сильном русскоязычном энкодере (rubert-base-cased).

Зачем обёртка: методика и весь код обучения уже проверены в v21 (`tools/ce_train_cpu_v21.py`,
`tools/ce_model_v21.py`). В v22 меняется только энкодер и место запуска:

  * весов rubert-base-cased в песочнице нет (HF/hf-mirror/ModelScope недоступны из песочницы,
    GitHub не хранит файлы > 100 МБ, LFS запрещён) → веса скачивает раннер GitHub Actions,
    там же идёт обучение (`artifacts/encoder_weights_v22.json`);
  * архитектура та же: `[CLS] doc_window [SEP] candidate [SEP]` + головы start/end/risk,
    CE как пара-детектор + пара-гейт поверх v11 (найдено в v21);
  * скрипт проксирует argv в тренер v21 и дописывает в отчёт сведения об энкодере
    (манифест весов, число параметров, режим прогона), чтобы артефакт v22 был самодостаточным.

Запуск (на раннере, после скачивания весов):
    python tools/ce_train_cpu_v22.py --encoder models/rubert-base-cased --arms real \
        --epochs 6 --batch 8 --lr 3e-5 --doc-tokens 96 --max-len 192 \
        --report artifacts/ce_train_v22.json --checkpoint artifacts/ce_model_v22.pt
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ce_train_cpu_v21 as V21


def _arg(argv, name, default=None):
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


def enrich(report_path, encoder_dir, checkpoint, started):
    report_path = Path(report_path)
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    encoder_dir = Path(encoder_dir) if encoder_dir else None
    manifest = {}
    if encoder_dir and (encoder_dir / "manifest.json").is_file():
        manifest = json.loads((encoder_dir / "manifest.json").read_text(encoding="utf-8"))
    parameters = {}
    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint and checkpoint.is_file():
        try:
            import torch

            state = torch.load(checkpoint, map_location="cpu")["state_dict"]
            total = int(sum(tensor.numel() for tensor in state.values()))
            parameters = {"checkpoint_bytes": checkpoint.stat().st_size,
                          "state_dict_parameters": total,
                          "fp32_mb": round(total * 4 / 1024 ** 2, 2)}
        except Exception as error:  # pragma: no cover - диагностика
            parameters = {"error": str(error)}
    run = {}
    run_path = ROOT / "artifacts/ce_run_v22.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
    report["version"] = "v22"
    report["rule"] = ("v22: тот же cross-encoder и та же методика, что в v21, но энкодер — "
                      "rubert-base-cased (178M параметров). Обучение идёт на раннере GitHub Actions "
                      "(весов в песочнице нет), метрики — только val и OOF по документам; held-out не читается")
    report["encoder"] = {"dir": str(encoder_dir) if encoder_dir else "models/rubert-tiny2",
                         "manifest": manifest, "parameters": parameters}
    report["run_mode"] = run
    report["wall_seconds_v22"] = round(time.time() - started, 1)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт v22 дополнен: {report_path}", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    report_path = _arg(argv, "--report", "artifacts/ce_train_v22.json")
    encoder_dir = _arg(argv, "--encoder", "")
    checkpoint = _arg(argv, "--checkpoint", "artifacts/ce_model_v22.pt")
    started = time.time()
    result = None
    try:
        result = V21.main(argv)
    except SystemExit as error:  # тренер v21 завершает работу через sys.exit в части веток
        if error.code not in (0, None):
            raise
    enrich(report_path, encoder_dir, checkpoint, started)
    return result


if __name__ == "__main__":
    main()
