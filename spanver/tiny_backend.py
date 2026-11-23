"""Бэкенд tiny: измеренная модель проекта (rubert-tiny2 + 5 лексических признаков).

Отличия от лексического модуля (`spanver/module.py`) и от ruSciBERT-бэкенда
(`spanver/model_score.py`):
  * вероятности на субтокен ответа даёт tiny2, поэтому границы точнее;
  * декодирование гистерезисное (start/extend), с gate по максимуму субтокенов
    и клаузным расширением — ровно то, что измерялось в CV проекта;
  * число фрагментов на ответ не ограничено единицей.

Всё необходимое уже лежит в пакете: `lexfeat.pair_probabilities` считает
признаки и вероятности, `decode` собирает спаны, `snap.snapped` расширяет их
по границам клауз, `module.predict_type` присваивает один из семи типов.
"""
import hashlib
import json
import tempfile
from pathlib import Path

from .decode import spans_from_flags
from .module import confidence, predict_type
from .snap import snapped

DEFAULT_WEIGHTS = "models/rubert-tiny2"
DEFAULT_HEAD = "artifacts/tiny_head.json"
DEFAULT_CHECKPOINT_DIR = "weights/final"
DEFAULT_CHECKPOINT_SHA256 = "04eb7932a33a7ed1a2f4e166f536b002ba3ad4a16e2a410a21e292e1558d747a"
TOP_CHUNKS = 4
MIN_CHARS = 2
DEFAULT_MAX_SPANS = 3


def head_path_is_explicit(path):
    return path not in (None, DEFAULT_HEAD)


class TinyUnavailable(RuntimeError):
    pass


class TinyHeadMissing(RuntimeError):
    pass


def load_head(path):
    target = Path(path)
    if not target.exists():
        raise TinyHeadMissing(f"нет конфигурации tiny-модели: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    for key in ("gate", "start", "extend", "top_chunks"):
        if key not in data:
            raise TinyHeadMissing(f"в {target} нет поля {key}")
    if data["start"] < data["extend"]:
        raise TinyHeadMissing("start должен быть не меньше extend")
    return data


def load_verified_checkpoint(model, directory, expected_sha256=DEFAULT_CHECKPOINT_SHA256):
    """Загрузить отдельный checkpoint после проверки манифеста, частей и SHA.

    Не используем tools: загрузка должна работать из поставляемого пакета.
    При отсутствии/повреждении весов ни в коем случае не используем
    случайную головку вместо измеренной модели.
    """
    import torch

    folder = Path(directory)
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "готово" or manifest.get("variant") != "v5" or not manifest.get("restore_verified"):
            raise ValueError("манифест не подтверждает восстановленный V5")
        entries = [item for item in manifest["checkpoints"] if item.get("seed") == 0]
        if len(entries) != 1:
            raise ValueError("в манифесте должен быть один checkpoint seed 0")
        entry = entries[0]
        if entry.get("sha256") != expected_sha256 or not entry.get("chunks"):
            raise ValueError("SHA checkpoint не соответствует ожидаемому V5")
        digest = hashlib.sha256()
        total = 0
        # Не оставляем собранный checkpoint на диске после загрузки.
        with tempfile.TemporaryFile() as target:
            for part in entry["chunks"]:
                name = part["name"]
                if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                    raise ValueError("некорректное имя части")
                chunk_digest = hashlib.sha256()
                chunk_size = 0
                with (folder / "seed_0" / name).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        chunk_digest.update(block)
                        digest.update(block)
                        chunk_size += len(block)
                        target.write(block)
                if chunk_size != part["bytes"] or chunk_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"повреждена часть checkpoint: {name}")
                total += chunk_size
            if total != entry["size"] or digest.hexdigest() != expected_sha256:
                raise ValueError("контрольная сумма или размер checkpoint не совпали")
            target.seek(0)
            state = torch.load(target, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        model.eval()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as error:
        raise TinyUnavailable(f"проверенный V5 seed 0 не загружен из {folder}: {error}") from error
    return expected_sha256


class TinyBundle:
    """Ленивая загрузка: torch и transformers нужны только в этом режиме."""

    def __init__(self, weights=None, head_path=DEFAULT_HEAD):
        # первый аргумент может быть путём к пресету, как у остальных бэкендов
        if weights and str(weights).endswith(".json") and not head_path_is_explicit(head_path):
            head_path, weights = weights, None
        self.head = load_head(head_path)
        self.directory = Path(weights or self.head.get("weights_dir") or DEFAULT_WEIGHTS)
        if not self.directory.exists():
            raise TinyUnavailable(f"каталог весов не найден: {self.directory}")
        try:
            from . import lexfeat
        except ImportError as error:  # pragma: no cover - зависит от окружения
            raise TinyUnavailable(f"для бэкенда tiny нужны torch и transformers: {error}") from error
        self.lexfeat = lexfeat
        self.tokenizer, self.model = lexfeat.load_feature_model(self.directory, labels=2, seed=int(self.head.get("seed", 0)))
        self.model.base.config._attn_implementation = "eager"
        self.checkpoint_dir = Path(self.head.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
        self.checkpoint_sha256 = load_verified_checkpoint(
            self.model, self.checkpoint_dir,
            self.head.get("checkpoint_sha256", DEFAULT_CHECKPOINT_SHA256),
        )
        self.model.eval()

    def probabilities(self, answer, document):
        pair = {"pair_id": "request", "model_output": answer or "", "document_text": document or ""}
        cached = self.lexfeat.pair_probabilities(self.tokenizer, self.model, pair, top=int(self.head["top_chunks"]))
        if cached is None:
            return []
        return cached["pieces"]

    def verify(self, answer, document, threshold=None, max_spans=None):
        text = answer or ""
        pieces = self.probabilities(text, document)
        gate = self.head["gate"] if threshold is None else threshold
        start = self.head["start"]
        extend = self.head["extend"]
        spans = []
        if pieces and max((piece[2] for piece in pieces), default=0.0) >= gate:
            flags = []
            active = False
            for _, _, score in pieces:
                active = score >= (extend if active else start)
                flags.append(active)
            spans = spans_from_flags(text, pieces, flags, MIN_CHARS)
        spans = snapped(spans, text, "expand")
        limit = DEFAULT_MAX_SPANS if max_spans is None else max_spans
        spans = sorted(spans, key=lambda item: (-item["risk"], item["start"]))[:limit]
        payload = []
        for span in spans:
            fragment = text[span["start"]: span["end"] + 1]
            payload.append({
                "start": span["start"],
                "end": span["end"],
                "risk": span["risk"],
                "type": predict_type(fragment, 0.0),
                "confidence": confidence(span["risk"]),
                "text": fragment,
            })
        covered = sum(len(item["text"]) for item in payload)
        ai_share = round(covered / len(text), 4) if text else 0.0
        return {
            "spans": payload,
            "ai_share": min(1.0, max(0.0, ai_share)),
            "meta": {
                "backend": "tiny",
                "weights": str(self.directory),
                "checkpoint_sha256": self.checkpoint_sha256,
                "gate": gate,
                "start": start,
                "extend": extend,
                "max_spans": limit,
                "clause_expand": True,
                "type_rule": "lexical seven-way",
                "ai_share_rule": "доля знаков ответа, покрытых фрагментами, без калибровки",
            },
        }
