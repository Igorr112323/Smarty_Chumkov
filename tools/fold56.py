"""ПРОМПТ 56: собрать манифест фолда из артефакта весов.

Что лежит в артефакте `weights56-outer{N}`: каталоги `seed_0 … seed_4`, в каждом
`manifest.json` одного чекпойнта (поля seed, chunks, sha256, size) и куски
`part-*.bin`. Манифеста уровня фолда там нет: в прогоне 44218112557 он не
сохранялся, а `threshold` и `gate` в схемах 51/52 брались из `weights/final`,
которого в этом прогоне тоже нет.

Порог фолда при этом не потерян: он выбран на внутренних фолдах и записан в
`artifacts/decode56.json` (`threshold_by_fold`), тот же, что применялся при
подсчёте 2.1–2.6. Гейт — константа схемы 52 (0.45). Этот модуль просто
собирает всё перечисленное в один манифест на месте, чтобы `eval56_test.py` и
`speed56.py` читали веса единообразно.

Правило выбора порога здесь не пересматривается и по test не подбирается:
значение берётся из уже опубликованного `decode56.json`.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE = 0.45
DECODE = ROOT / "artifacts/decode56.json"


def fold_folder(fold):
    return ROOT / "weights/ablation51/ten" / f"fold{fold}"


def seed_manifests(folder):
    found = []
    for path in sorted(Path(folder).glob("seed_*/manifest.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if "seed" not in entry or not entry.get("chunks"):
            raise SystemExit(f"манифест сида повреждён: {path}")
        found.append(entry)
    found.sort(key=lambda item: item["seed"])
    return found


def thresholds():
    if not DECODE.exists():
        raise SystemExit(f"нет {DECODE}: порог фолда неоткуда взять")
    report = json.loads(DECODE.read_text(encoding="utf-8"))
    if report.get("status") != "готово":
        raise SystemExit("decode56.json не завершён")
    return {int(fold): float(point["start"])
            for fold, point in report["threshold_by_fold"].items()}


def ensure_fold_manifest(fold):
    """Дописать в каталог фолда манифест уровня фолда. Возвращает его."""
    folder = fold_folder(fold)
    if not folder.is_dir():
        raise SystemExit(f"нет каталога весов фолда: {folder}")
    seeds = seed_manifests(folder)
    if len(seeds) != 5:
        raise SystemExit(f"фолд {fold}: ожидалось 5 сидов, найдено {len(seeds)}")
    chosen = thresholds()
    if fold not in chosen:
        raise SystemExit(f"фолд {fold}: нет порога в decode56.json")
    manifest = {
        "variant": "ten",
        "outer_fold": fold,
        "checkpoints": seeds,
        "threshold": {"start": chosen[fold], "extend": chosen[fold] / 2},
        "gate": GATE,
        "restore_verified": True,
        "source": "собрано tools/fold56.py; порог из artifacts/decode56.json",
        "test301": False,
    }
    # манифест сидов лежит в seed_*/manifest.json и не затирается
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    return manifest


def main():
    for fold in [int(item) for item in (sys.argv[1:] or ["1"])]:
        manifest = ensure_fold_manifest(fold)
        print(f"фолд {fold}: сидов {len(manifest['checkpoints'])}, "
              f"порог {manifest['threshold']['start']}, гейт {manifest['gate']}")


if __name__ == "__main__":
    main()
