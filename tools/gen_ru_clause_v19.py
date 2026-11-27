#!/usr/bin/env python3
"""Programmatic Russian clause-level hallucination generator (v19).

Generates synthetic Russian (document, answer) pairs with realistic clause-level
hallucinations (whole / clause / short) strictly from local non-RusHallu-RAG
corpora, using cross-corpus full-sentence clause insertion (zero entity/number
flips, zero manual annotation, doc_overlap == 0 with RusHallu-RAG).

Usage:
    python tools/gen_ru_clause_v19.py \
        --dataset Dataset \
        --out-jsonl artifacts/ru_clause_gen_v19.jsonl \
        --out-report artifacts/ru_clause_gen_v19.json \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.build import build_gold
from spanver.eval_report import ceiling as eval_ceiling
from spanver.evaluate import evaluate_pairs, iou, metrics, token_positions
from spanver.module import (
    IOU_THRESHOLD,
    STOPWORDS,
    candidate_spans,
    ngram_set,
    sentence_spans,
    stem,
    tokens,
)
from spanver.pairs import doc_ids as spanver_doc_ids
from spanver.qc import inspect as qc_inspect
from spanver.snap import snapped
from spanver.types import TYPE_KEYS, classify

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


# ---------------------------------------------------------------------------
# Text normalization & Russian sentence segmentation
# ---------------------------------------------------------------------------

ABBREV_RE = re.compile(
    r"\b(?:т\.?\s*д|т\.?\s*п|т\.?\s*е|т\.?\s*к|т\.?\s*н|и\s+др|и\s+пр|др|пр|см|ср|"
    r"стр|рис|табл|им|ул|кв|обл|тыс|млн|млрд|руб|коп|проф|акад|доц|канд|ст|гл|разд|"
    r"вып|изд|пер|ред|сост|авт|сб|н\.\s*э|ок|прим|напр|гг|вв|г|в|д|ч|п|с)\.$",
    re.I,
)
INITIAL_RE = re.compile(r"(?:^|[\s(«\"])[А-ЯЁA-Z]\.$")
LIST_NUM_END_RE = re.compile(r"(?:^|[\s:])\d{1,2}\.$")

DANGLING_START_RE = re.compile(
    r"^(?:Он|Она|Оно|Они|Его|Её|Ее|Их|Ему|Ей|Им|Нем|Нём|Ней|Них|"
    r"Эт(?:от|а|о|и|у|ой|им|их|ого|ому|ом)|"
    r"Т(?:от|а|о|е|у|ой|ем|ех|ого|ому|ом)|"
    r"Так(?:ой|ая|ое|ие|ую|ого|ому|им|их|ими)|"
    r"Данн(?:ый|ая|ое|ые|ую|ого|ой|ому|ым|ых|ыми)|"
    r"Указанн(?:ый|ая|ое|ые|ую|ого|ой|ым|ых)|"
    r"Подобн(?:ый|ая|ое|ые|ую|ого|ой|ым|ых)|"
    r"Также|Однако|Поэтому|При этом|Кроме|Помимо|Более того|Вместе с тем|Тем не менее|"
    r"Несмотря на|Вопреки|Затем|Потом|После этого|В результате|В итоге|В частности|"
    r"Например|То есть|Иначе говоря|Таким образом|Но|А|И|Или|Хотя|Потому|Поскольку|"
    r"Если|Ведь|Из них|Среди них|Оттуда|Отсюда|Здесь|Там|Тогда|Сейчас|Теперь|"
    r"Ранее|Позже|Впоследствии|Наконец|Во-первых|Во-вторых|В-третьих|В связи с этим|"
    r"В свою очередь|С другой стороны|Напротив|Наоборот|Все они|Оба|Обе|"
    r"Некоторые из них|Часть из них|Большинство из них|Выше|Ниже|Далее|"
    r"Следовательно|Значит|Между тем|Впрочем|Правда|Конечно|Действительно|"
    r"Всего|Другой|Другая|Другое|Другие|Иной|Иная|Иное|Иные|Остальные|Остальная|"
    r"Остальное|Последний|Последняя|Последнее|Последние|Предыдущий|Следующий|"
    r"Словом|Итак|Да|Нет|Короче|Вообще|Было показано|Как отмечалось|Как упоминалось|"
    r"Как сообщалось|В то же время|В то время как|Между прочим)\b"
)

ANAPHORA_NEXT_RE = re.compile(
    r"^(?:Он|Она|Оно|Они|Его|Её|Ее|Их|Ему|Ей|Им|Этот|Эта|Это|Эти|Тот|Та|То|Те|"
    r"Такой|Такая|Такое|Такие|Данный|Данная|Данное|Данные|Указанный|Указанная|"
    r"Подобный|Подобная|Он же|Она же|Они же|Из них|Среди них|Кроме того|При этом|"
    r"Однако|Поэтому|Затем|Потом|После этого)\b"
)

EARLY_ANAPHORA_RE = re.compile(
    r"\b(?:он|она|оно|они|его|её|ее|их|ему|ей|им|них|ним|ней|"
    r"этого|этой|этим|этих|этот|эта|эти|эту|"
    r"данный|данная|данное|данные|данной|данного|данном|данным|данных|"
    r"такого|такой|такую|таким|таких|такими|указанный|указанной|указанных)\b",
    re.I,
)

SUBJECTLESS_YEAR_RE = re.compile(
    r"^(?:В|С)\s+\d{4}(?:[—-]\d{4})?\s+год(?:у|а|ах|ов)?\s+"
    r"(?:был|была|было|были|стал|стала|стало|стали|мог|могла|имел|имела|вёл|вел|шёл|шел|"
    r"[а-яё]{1,}(?:ал|ала|али|ил|ила|или|ел|ела|ели|ул|ула|ули|ся|сь))\b"
)

SUBJECTLESS_START_RE = re.compile(
    r"^(?:Имеет|Имеют|Имел|Имела|Имели|Не\s+имеет|Не\s+является|Не\s+может|"
    r"Расположен|Расположена|Расположено|Расположены|Находится|Находятся|"
    r"Состоит|Состоят|Является|Являются|Относится|Относятся|"
    r"Используется|Используются|Применяется|Применяются|Считается|Считаются|"
    r"Включает|Включают|Содержит|Содержат|Обладает|Обладают|Отличается|Отличаются|"
    r"Представляет|Представляют|Называется|Называются|Встречается|Встречаются|"
    r"Легко\s+\w+[аяе]тся|Иногда\s+\w+[аяею]т|К\s+тому\s+же|К\s+счастью|К\s+сожалению)\b"
)

FIRST_SECOND_PERSON_RE = re.compile(
    r"(?:^|[^\wА-Яа-яЁё])(?:я|мне|меня|мной|мною|мой|моя|моё|мое|мои|моего|моей|моим|"
    r"мы|нам|нас|нами|наш|наша|наше|наши|нашего|нашей|ты|тебе|тебя|тобой|твой|твоя|"
    r"твоё|твое|вы|вам|вас|вами|ваш|ваша|ваше|привет|спасибо|пожалуйста|кретин|"
    r"дурак|идиот|лол|ахах|кстати|короче|блин|чёрт|черт|хрен|фиг|божка|дубасят)(?=$|[^\wА-Яа-яЁё])",
    re.I,
)

HEADER_START_RE = re.compile(
    r"^(?:Цель|Методы|Материал|Материалы|Результаты|Выводы|Заключение|Введение|"
    r"Актуальность|Обсуждение|Ключевые слова|Таблица|Рисунок|Список|Примечания|"
    r"См\.|Источник|Отрывок|Фрагмент|Пассаж|Проход|Вопрос|Ответ|"
    r"В данной статье|В данной работе|В настоящей работе|В этом исследовании|"
    r"В нашем исследовании|Вот краткое|Вот краткий|Вот обзор|Вот объективный|Вот резюме|"
    r"Ниже приведено|Ниже представлен)\b",
    re.I,
)

WRAPPER_PREFIX_RE = re.compile(
    r"^(?:Вот\s+(?:кратк\w+|объективн\w+|резюме|ответ|обзор|сводк\w+|информац\w+)[^:\n]{0,160}:\s*|"
    r"На\s+основании\s+предоставленных\s+[^:\n]{0,120}:\s*|"
    r"Конечно!\s*Вот[^:\n]{0,160}:\s*)",
    re.I,
)

NUMBERED_LIST_RE = re.compile(r"(?:^|\s)1\.\s+.*?\s2\.\s+")

EXPLICIT_PREDICATE_RE = re.compile(
    r"(?:^|\s)(?:является|являются|являлся|являлась|являлось|являлись|был|была|было|были|"
    r"будет|будут|стал|стала|стало|стали|становится|становятся|может|могут|имеет|имеют|"
    r"имел|имела|имели|содержит|содержат|состоит|состоят|включает|включают|входит|входят|"
    r"используется|используются|находится|находятся|расположен|расположена|расположено|"
    r"расположены|происходит|происходят|существует|существуют|позволяет|позволяют|"
    r"обеспечивает|обеспечивают|приводит|приводят|составляет|составляют|отмечается|"
    r"отмечаются|наблюдается|наблюдаются|характеризуется|характеризуются|представляет|"
    r"представляют|называется|называются|считается|считаются|относится|относятся|"
    r"остаётся|остается|появился|появилась|появилось|появились|начал|начала|начало|начали|"
    r"начался|началась|началось|начались|состоялся|состоялась|состоялось|состоялись|"
    r"получил|получила|получили|прошёл|прошла|прошло|прошли|достиг|достигла|достигли|"
    r"вырос|выросла|выросли|снизился|снизилась|увеличился|увеличилась|уменьшился|уменьшилась|"
    r"проводится|проводятся|изучается|изучаются|применяется|применяются|формируется|"
    r"формируются|создан|создана|создано|созданы|основан|основана|основано|основаны|"
    r"построен|построена|построено|построены|открыт|открыта|открыто|открыты|принят|"
    r"принята|принято|приняты|выпущен|выпущена|выпущены|введён|введена|введено|введены|"
    r"утверждён|утверждена|установлен|установлена|установлено|установлены|выявлен|"
    r"выявлена|выявлено|выявлены|обнаружен|обнаружена|обнаружено|обнаружены|показан|"
    r"показана|показано|показаны|подтверждён|подтверждена|подтверждено|доказан|доказана|"
    r"доказано|запрещён|запрещена|запрещено|разрешён|разрешена|разработан|разработана|"
    r"разработано|разработаны)(?=$|[^\wА-Яа-яЁё])",
    re.I,
)

VERB_SUFFIX_RE = re.compile(
    r"\b[а-яё]{3,}(?:ает|яет|ует|юет|ают|яют|уют|юют|ится|ятся|ется|ются|"
    r"ались|ились|елись|улись|ался|ился|елся|улся|алась|илась|елась|улась|"
    r"алось|илось|елось|улось|овал|овала|овало|овали|ировал|ировала|ировали|"
    r"ирует|ируют|ляет|ляют|няет|няют|щает|щают|шает|шают|жает|жают|вает|вают|"
    r"дает|дают|тает|тают|одит|одят|озит|озят|осит|осят|оит|оят|авит|авят|елит|елят)\b"
)

INFOBOX_RUN_RE = re.compile(r"(?:[А-ЯЁ][а-яё]+\s+){4,}[А-ЯЁ][а-яё]+")
SHORT_DANGLING_INTERNAL_RE = re.compile(
    r"\b(?:он|она|оно|они|его|её|ее|их|ему|ей|им|этом|этой|этого|этих|такой|такая|такое|"
    r"иная|другая|иной|другой|других|другом|также|ещё|еще|уже|тоже|здесь|там|тогда)\b",
    re.I,
)
META_REF_RE = re.compile(
    r"(?:вышеуказанн|нижеследующ|данной стать|данной работ|настоящей работ|"
    r"этом исследован|отрывке\s*\d|passage\s*\d|в\s+тексте\s+указано|на\s+основании\s+предоставленн|"
    r"что\s+это\b|\bсм\.|\bтабл\.|\bрис\.|РИА\s+Новости|\bТАСС\b|Интерфакс)",
    re.I,
)
FORBIDDEN_CHARS = set("{}[]<>|*#@/\\=_~^`")
PROPER_NOUN_RE = re.compile(r"(?<!^)(?<![.!?]\s)[А-ЯЁA-Z][а-яёa-z]{3,}")
NUMBER_TOKEN_RE = re.compile(r"\b\d{2,}\b")
QA_PASSAGE_KEYS = ("отрывки", "проходы", "пассажи", "passages", "отрывок", "переходы", "Проходы")


def strip_control_chars(text: str) -> str:
    return "".join(
        ch
        for ch in (text or "")
        if not (unicodedata.category(ch) == "Cc" and ch not in "\n\r\t")
    )


def clean_whitespace(text: str) -> str:
    cleaned = strip_control_chars(text).replace("\u0301", "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", strip_control_chars(text).replace("\u0301", "").strip().lower())


def sha256_text(text: str) -> str:
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def cyr_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text or "")
    if not letters:
        return 0.0
    cyr = re.findall(r"[А-Яа-яЁё]", text or "")
    return len(cyr) / len(letters)


def word_list(text: str) -> list[str]:
    return re.findall(r"[0-9а-яёa-z]+", norm_text(text))


def word_ngrams(text: str, n: int = 8) -> set[str]:
    ws = word_list(text)
    if len(ws) < n:
        return set()
    return {" ".join(ws[i : i + n]) for i in range(len(ws) - n + 1)}


def content_stems(text: str) -> set[str]:
    return {
        stem(tok)
        for tok in tokens(text)
        if tok.lower() not in STOPWORDS and len(tok) >= 3
    }


def split_ru_sentences(text: str) -> list[str]:
    cleaned = clean_whitespace(text)
    if not cleaned:
        return []
    sents: list[str] = []
    start = 0
    for match in re.finditer(r"([.!?]+)\s+(?=[А-ЯЁ«\"(\d])", cleaned):
        end = match.end(1)
        cand = cleaned[start:end].strip()
        if ABBREV_RE.search(cand) or INITIAL_RE.search(cand) or LIST_NUM_END_RE.search(cand):
            continue
        sents.append(cand)
        start = match.end()
    tail = cleaned[start:].strip()
    if tail:
        sents.append(tail)
    return sents


def is_clean_base_sentence(sentence: str) -> bool:
    if not sentence or not sentence.endswith((".", "!", "?")):
        return False
    if not re.match(r"^[А-ЯЁ«\"]", sentence):
        return False
    if len(sentence) < 20:
        return False
    if cyr_ratio(sentence) < 0.85:
        return False
    if any(ch in FORBIDDEN_CHARS for ch in sentence):
        return False
    if re.search(r"[а-яё]-\s+[а-яё]", sentence):
        return False
    if re.search(r"\.\s+\.", sentence):
        return False
    if sentence.count("(") != sentence.count(")") or sentence.count("«") != sentence.count("»"):
        return False
    if sentence.count('"') % 2 != 0:
        return False
    return True


def is_clean_donor_sentence(sentence: str, is_short: bool = False) -> bool:
    if not sentence or not sentence.endswith("."):
        return False
    if not re.match(r"^[А-ЯЁ]", sentence):
        return False
    if re.match(r"^[А-ЯЁ]\.\s", sentence):  # broken initial start ("М. Степняк...")
        return False
    if re.match(r"^[А-ЯЁ]{2,6}\b", sentence):  # unexplained uppercase acronym start
        return False
    if re.match(r"^[А-ЯЁ]{3,}\s*,", sentence):  # news wire dateline ("ХАБАРОВСК, ...")
        return False
    if re.search(r"\s[.,;:!?]", sentence):
        return False
    if re.search(r"[,;:][А-Яа-яЁё0-9]", sentence):
        return False
    if re.search(r"[а-яё]-\s+[а-яё]", sentence):
        return False
    if re.search(r"\.\s+\.", sentence):
        return False
    if re.search(r"\b\d+\.\s+[a-hа-я]", sentence):  # chess/inline numbered notation ("1. e4")
        return False
    if any(ch in FORBIDDEN_CHARS for ch in sentence):
        return False
    low = sentence.lower()
    if "http" in low or "www." in low or "«»" in sentence or '""' in sentence:
        return False
    if sentence.count("(") != sentence.count(")") or sentence.count("«") != sentence.count("»"):
        return False
    if sentence.count('"') % 2 != 0:
        return False
    if cyr_ratio(sentence) < 0.88:
        return False
    if ABBREV_RE.search(sentence):
        return False
    if DANGLING_START_RE.match(sentence):
        return False
    if SUBJECTLESS_YEAR_RE.match(sentence):
        return False
    if SUBJECTLESS_START_RE.match(sentence):
        return False
    if FIRST_SECOND_PERSON_RE.search(sentence):
        return False
    if HEADER_START_RE.match(sentence):
        return False
    if INFOBOX_RUN_RE.search(sentence):
        return False
    if "— уже" in sentence or "- уже" in sentence:
        return False
    if ":" in sentence[:25]:
        return False
    if META_REF_RE.search(sentence):
        return False
    # Avoid dense chains of personal initials ("Ф. И. ..., А. Н. ...") that break naive sentence regexes
    if len(re.findall(r"\b[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.", sentence)) >= 2:
        return False
    # Reject dangling pronouns/demonstratives in the opening clause (first 55 chars)
    head = sentence[:55]
    if EARLY_ANAPHORA_RE.search(head):
        return False
    words = re.findall(r"[А-Яа-яЁё0-9A-Za-z-]+", sentence)
    min_words = 4 if is_short else 6
    if len(words) < min_words:
        return False
    if words[0].lower() == words[1].lower():
        return False
    if is_short and SHORT_DANGLING_INTERNAL_RE.search(sentence):
        return False
    has_pred = bool(
        EXPLICIT_PREDICATE_RE.search(sentence)
        or VERB_SUFFIX_RE.search(sentence)
        or " — " in sentence
    )
    return has_pred


def max_contiguous_word_overlap(clause: str, reference: str) -> int:
    cw = word_list(clause)
    rw = word_list(reference)
    if not cw or not rw:
        return 0
    pos: dict[str, list[int]] = defaultdict(list)
    for idx, w in enumerate(rw):
        pos[w].append(idx)
    best = 0
    for i, w in enumerate(cw):
        for j in pos.get(w, ()):
            k = 1
            while i + k < len(cw) and j + k < len(rw) and cw[i + k] == rw[j + k]:
                k += 1
            if k > best:
                best = k
    return best


def sentence_support_stats(sents: list[str], document: str) -> tuple[float, float]:
    dstems = content_stems(document)
    if not dstems:
        return 0.0, 0.0
    sups: list[float] = []
    for s in sents:
        ss = content_stems(s)
        if not ss:
            continue
        sups.append(len(ss & dstems) / len(ss))
    if not sups:
        return 0.0, 0.0
    return min(sups), statistics.mean(sups)


# ---------------------------------------------------------------------------
# RusHallu-RAG Exclusion / Overlap Index (strict doc_overlap == 0)
# ---------------------------------------------------------------------------

class RusHalluBlacklist:
    """Strict multi-level exclusion index built from RusHallu-RAG (gold + unlabeled)."""

    def __init__(self, dataset_dir: Path) -> None:
        self.gold_pairs = build_gold(dataset_dir, {})
        self.full_hashes: set[str] = set()
        self.subdoc_hashes: set[str] = set()
        self.doc_id_set: set[str] = set()
        self.prefixes: set[str] = set()
        self.ngrams8: set[str] = set()
        self.sent_hashes: set[str] = set()
        self.gold_doc_count = 0
        self.unlabeled_doc_count = 0

        for pair in self.gold_pairs:
            self._index_document(pair.get("document_text") or "")
            self.gold_doc_count += 1

        unlabeled_path = dataset_dir / "2_rushallu_rag_pairs" / "sberquad-train-rag.parquet"
        if unlabeled_path.exists() and pq is not None:
            for row in pq.read_table(unlabeled_path).to_pylist():
                self._index_document(row.get("joined_reference") or "")
                self.unlabeled_doc_count += 1

    def _index_document(self, doc: str) -> None:
        if not doc:
            return
        self.full_hashes.add(sha256_text(doc))
        for did in spanver_doc_ids(doc):
            self.doc_id_set.add(did)
        parts = re.split(r"(?m)^##[ \t]+.+?$", doc)
        for part in parts:
            part = part.strip()
            if len(part) >= 30:
                self.subdoc_hashes.add(sha256_text(part))
                self.prefixes.add(norm_text(part)[:100])
                self.ngrams8.update(word_ngrams(part, 8))
                for s in split_ru_sentences(part):
                    if len(s) >= 25:
                        self.sent_hashes.add(sha256_text(s))

    def overlaps_document(self, doc: str) -> bool:
        if not doc:
            return True
        nt = norm_text(doc)
        h = hashlib.sha256(nt.encode("utf-8")).hexdigest()
        if h in self.full_hashes or h in self.subdoc_hashes:
            return True
        for did in spanver_doc_ids(doc):
            if did in self.doc_id_set:
                return True
        if len(nt) >= 30 and nt[:100] in self.prefixes:
            return True
        ng = word_ngrams(doc, 8)
        if ng and (ng & self.ngrams8):
            return True
        return False

    def overlaps_clause(self, clause: str) -> bool:
        if not clause:
            return True
        if sha256_text(clause) in self.sent_hashes:
            return True
        ng = word_ngrams(clause, 8)
        if ng and (ng & self.ngrams8):
            return True
        return False


# ---------------------------------------------------------------------------
# Local corpus loaders (non-RusHallu-RAG ONLY)
# ---------------------------------------------------------------------------

def extract_clean_prefix(ans_raw: str, max_chars: int = 560, min_sents: int = 2) -> list[str] | None:
    """Extract a clean contiguous prefix of Russian sentences without skipping leading/middle ones."""
    cleaned = WRAPPER_PREFIX_RE.sub("", clean_whitespace(ans_raw)).strip()
    if not cleaned or NUMBERED_LIST_RE.search(cleaned):
        return None
    if FIRST_SECOND_PERSON_RE.search(cleaned):
        return None
    sents = split_ru_sentences(cleaned)
    if len(sents) < min_sents:
        return None
    if DANGLING_START_RE.match(sents[0]):
        return None
    chosen: list[str] = []
    for s in sents:
        if not is_clean_base_sentence(s):
            break
        if chosen and len(" ".join(chosen + [s])) > max_chars and len(chosen) >= min_sents:
            break
        chosen.append(s)
    if len(chosen) < min_sents:
        return None
    return chosen


def load_local_corpora(dataset_dir: Path, blacklist: RusHalluBlacklist) -> dict:
    """Load clean base pairs and donor pools from local non-RusHallu-RAG corpora."""
    audit: dict[str, dict] = {}
    base_multi: list[dict] = []
    base_single_sci: list[dict] = []
    seen_base_doc_hashes: set[str] = set()

    donors_short: dict[str, list[str]] = defaultdict(list)
    donors_clause: dict[str, list[str]] = defaultdict(list)
    donors_whole: dict[str, list[str]] = defaultdict(list)
    seen_donor_sent: set[str] = set()
    seen_donor_whole: set[str] = set()

    def harvest_donors(corpus: str, doc_text: str) -> None:
        sents = split_ru_sentences(doc_text)
        for idx, s in enumerate(sents):
            sh = sha256_text(s)
            if sh in seen_donor_sent or blacklist.overlaps_clause(s):
                continue
            if 20 <= len(s) <= 39 and is_clean_donor_sentence(s, is_short=True):
                seen_donor_sent.add(sh)
                donors_short[corpus].append(s)
            elif 45 <= len(s) <= 280 and is_clean_donor_sentence(s, is_short=False):
                seen_donor_sent.add(sh)
                donors_clause[corpus].append(s)
                if 115 <= len(s) <= 320 and sh not in seen_donor_whole:
                    seen_donor_whole.add(sh)
                    donors_whole[corpus].append(s)
                if idx + 1 < len(sents):
                    s2 = sents[idx + 1]
                    if (
                        45 <= len(s2) <= 240
                        and is_clean_donor_sentence(s2, is_short=False)
                        and not blacklist.overlaps_clause(s2)
                    ):
                        block = s + " " + s2
                        bh = sha256_text(block)
                        if 140 <= len(block) <= 440 and bh not in seen_donor_whole:
                            seen_donor_whole.add(bh)
                            donors_whole[corpus].append(block)

    # 1. RAGTruth_ru (negative for clean bases + prose docs from neg/pos for donors)
    rt_neg_path = dataset_dir / "RAGTruth_ru" / "ragtruth_ru_negative.jsonl"
    rt_pos_path = dataset_dir / "RAGTruth_ru" / "ragtruth_ru.jsonl"
    rt_stats = {"present": rt_neg_path.exists(), "raw_rows": 0, "rejected_overlap": 0, "clean_bases": 0, "donor_docs": 0}
    seen_rt_donor_docs: set[str] = set()
    if rt_neg_path.exists():
        with open(rt_neg_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rt_stats["raw_rows"] += 1
                row = json.loads(line)
                d_raw = (row.get("document_text") or "").strip()
                ans_raw = row.get("model_output") or ""
                is_qa = False
                if d_raw.startswith("{"):
                    try:
                        obj = json.loads(d_raw)
                    except json.JSONDecodeError:
                        continue
                    doc = ""
                    for key in QA_PASSAGE_KEYS:
                        if key in obj and isinstance(obj[key], str):
                            doc = clean_whitespace(obj[key])
                            is_qa = True
                            break
                    if not doc:
                        continue
                else:
                    doc = clean_whitespace(d_raw)
                if blacklist.overlaps_document(doc):
                    rt_stats["rejected_overlap"] += 1
                    continue
                dh = sha256_text(doc)
                if not is_qa and dh not in seen_rt_donor_docs and cyr_ratio(doc) >= 0.85:
                    seen_rt_donor_docs.add(dh)
                    rt_stats["donor_docs"] += 1
                    harvest_donors("ragtruth_ru", doc)
                if dh in seen_base_doc_hashes:
                    continue
                if cyr_ratio(doc) < 0.82 or len(doc) < 150:
                    continue
                chosen = extract_clean_prefix(ans_raw, max_chars=560, min_sents=2)
                if not chosen:
                    continue
                min_sup, mean_sup = sentence_support_stats(chosen, doc)
                if min_sup < 0.25 or mean_sup < 0.45:
                    continue
                seen_base_doc_hashes.add(dh)
                rt_stats["clean_bases"] += 1
                base_multi.append(
                    {
                        "base_corpus": "ragtruth_ru",
                        "domain": "qa" if is_qa else "summary",
                        "base_id": str(row.get("pair_id") or ""),
                        "document_text": doc,
                        "doc_hash": dh,
                        "sents": chosen,
                        "min_support": round(min_sup, 4),
                        "mean_support": round(mean_sup, 4),
                    }
                )
    if rt_pos_path.exists():
        with open(rt_pos_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                d_raw = (row.get("document_text") or "").strip()
                if d_raw.startswith("{"):
                    continue
                doc = clean_whitespace(d_raw)
                dh = sha256_text(doc)
                if dh in seen_rt_donor_docs or blacklist.overlaps_document(doc):
                    continue
                if cyr_ratio(doc) >= 0.85:
                    seen_rt_donor_docs.add(dh)
                    rt_stats["donor_docs"] += 1
                    harvest_donors("ragtruth_ru", doc)
    audit["ragtruth_ru"] = rt_stats

    # 2. RuFacts (label == 0 for clean bases; unique evidences for donors)
    rf_paths = [
        dataset_dir / "RuFacts" / "train_with_spans.json",
        dataset_dir / "RuFacts" / "val_with_spans.json",
    ]
    rf_stats = {"present": any(p.exists() for p in rf_paths), "raw_rows": 0, "rejected_overlap": 0, "clean_bases": 0, "donor_docs": 0}
    seen_rf_donor_docs: set[str] = set()
    for path in rf_paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rf_stats["raw_rows"] += 1
                row = json.loads(line)
                doc = clean_whitespace(row.get("evidence") or "")
                ans_raw = row.get("claim") or ""
                if blacklist.overlaps_document(doc):
                    rf_stats["rejected_overlap"] += 1
                    continue
                dh = sha256_text(doc)
                if dh not in seen_rf_donor_docs and cyr_ratio(doc) >= 0.88:
                    seen_rf_donor_docs.add(dh)
                    rf_stats["donor_docs"] += 1
                    harvest_donors("rufacts", doc)
                if int(row.get("label", 1)) != 0:
                    continue
                if dh in seen_base_doc_hashes:
                    continue
                if cyr_ratio(doc) < 0.85 or len(doc) < 120 or not re.match(r"^[А-ЯЁ«\"]", doc):
                    continue
                if FIRST_SECOND_PERSON_RE.search(doc):
                    continue
                chosen = extract_clean_prefix(ans_raw, max_chars=560, min_sents=2)
                if not chosen or len(" ".join(chosen)) < 80:
                    continue
                min_sup, mean_sup = sentence_support_stats(chosen, doc)
                if min_sup < 0.25 or mean_sup < 0.45:
                    continue
                seen_base_doc_hashes.add(dh)
                rf_stats["clean_bases"] += 1
                base_multi.append(
                    {
                        "base_corpus": "rufacts",
                        "domain": "news_fact",
                        "base_id": f"rufacts_{row.get('idx', '')}",
                        "document_text": doc,
                        "doc_hash": dh,
                        "sents": chosen,
                        "min_support": round(min_sup, 4),
                        "mean_support": round(mean_sup, 4),
                    }
                )
    audit["rufacts"] = rf_stats

    # 3. mFAVA_ru (Russian subset: contiguous untagged runs for clean bases; references for donors)
    mfava_path = dataset_dir / "mFAVA_ru" / "test.json"
    mf_stats = {"present": mfava_path.exists(), "raw_rows": 0, "rejected_overlap": 0, "clean_bases": 0, "donor_docs": 0}
    if mfava_path.exists():
        rows = json.loads(mfava_path.read_text(encoding="utf-8"))
        for idx, row in enumerate(rows):
            if row.get("language") != "russian":
                continue
            mf_stats["raw_rows"] += 1
            doc = clean_whitespace(row.get("references") or "")
            ga = (row.get("gold_annotations") or "").strip()
            if blacklist.overlaps_document(doc):
                mf_stats["rejected_overlap"] += 1
                continue
            dh = sha256_text(doc)
            if cyr_ratio(doc) >= 0.82:
                mf_stats["donor_docs"] += 1
                harvest_donors("mfava_ru", doc)
            if dh in seen_base_doc_hashes or cyr_ratio(doc) < 0.80:
                continue
            parts = re.split(r"(</?[a-z_]+>)", ga, flags=re.I)
            depth = 0
            chunks: list[str] = []
            for part in parts:
                if re.fullmatch(r"<[a-z_]+>", part, re.I):
                    depth += 1
                    chunks.append("<<<TAG>>>")
                elif re.fullmatch(r"</[a-z_]+>", part, re.I):
                    depth = max(0, depth - 1)
                    chunks.append("<<<TAG>>>")
                else:
                    chunks.append(part if depth == 0 else "<<<TAG>>>")
            marked = clean_whitespace("".join(chunks))
            # Find the longest contiguous run of clean untagged sentences starting without dangling anaphora
            all_s = split_ru_sentences(marked)
            best_run: list[str] = []
            cur_run: list[str] = []
            for s in all_s:
                if "<<<TAG>>>" in s or not is_clean_base_sentence(s) or not s.endswith("."):
                    if len(cur_run) > len(best_run):
                        best_run = cur_run
                    cur_run = []
                    continue
                if not cur_run and DANGLING_START_RE.match(s):
                    continue
                cur_run.append(s)
            if len(cur_run) > len(best_run):
                best_run = cur_run
            if len(best_run) < 2:
                continue
            chosen = []
            for s in best_run:
                if chosen and len(" ".join(chosen + [s])) > 560 and len(chosen) >= 2:
                    break
                chosen.append(s)
            if len(chosen) < 2:
                continue
            min_sup, mean_sup = sentence_support_stats(chosen, doc)
            if min_sup < 0.25 or mean_sup < 0.45:
                continue
            seen_base_doc_hashes.add(dh)
            mf_stats["clean_bases"] += 1
            base_multi.append(
                {
                    "base_corpus": "mfava_ru",
                    "domain": "wikipedia",
                    "base_id": f"mfava_ru_{idx}",
                    "document_text": doc,
                    "doc_hash": dh,
                    "sents": chosen,
                    "min_support": round(min_sup, 4),
                    "mean_support": round(mean_sup, 4),
                }
            )
    audit["mfava_ru"] = mf_stats

    # 4. ruSciFact (scientific abstracts & label == 0 clean claims; strictly disjoint from RusHallu-RAG)
    rsf_paths = [
        dataset_dir / "5_ruscifact" / "ruscifact_train.parquet",
        dataset_dir / "5_ruscifact" / "ruscifact_val.parquet",
    ]
    rsf_stats = {"present": any(p.exists() for p in rsf_paths), "raw_rows": 0, "rejected_overlap": 0, "clean_bases_multi": 0, "clean_bases_single": 0, "donor_docs": 0}
    seen_rsf_donor_docs: set[str] = set()
    if pq is not None:
        for path in rsf_paths:
            if not path.exists():
                continue
            for row in pq.read_table(path).to_pylist():
                rsf_stats["raw_rows"] += 1
                doc = clean_whitespace(row.get("ru_abstract") or "")
                ans_raw = clean_whitespace(row.get("ruscifact") or "").strip("`")
                if blacklist.overlaps_document(doc):
                    rsf_stats["rejected_overlap"] += 1
                    continue
                dh = sha256_text(doc)
                if dh not in seen_rsf_donor_docs and cyr_ratio(doc) >= 0.80:
                    seen_rsf_donor_docs.add(dh)
                    rsf_stats["donor_docs"] += 1
                    harvest_donors("ruscifact", doc)
                if str(row.get("label")) != "0":
                    continue
                if dh in seen_base_doc_hashes or cyr_ratio(doc) < 0.80 or cyr_ratio(ans_raw) < 0.85:
                    continue
                sents = [s for s in split_ru_sentences(ans_raw) if is_clean_base_sentence(s) and s.endswith(".")]
                if not sents or len(sents) != len(split_ru_sentences(ans_raw)):
                    continue
                if DANGLING_START_RE.match(sents[0]):
                    continue
                min_sup, mean_sup = sentence_support_stats(sents, doc)
                if min_sup < 0.25 or mean_sup < 0.40:
                    continue
                seen_base_doc_hashes.add(dh)
                entry = {
                    "base_corpus": "ruscifact",
                    "domain": "scientific",
                    "base_id": f"ruscifact_{row.get('@id', '')}",
                    "document_text": doc,
                    "doc_hash": dh,
                    "sents": sents,
                    "min_support": round(min_sup, 4),
                    "mean_support": round(mean_sup, 4),
                }
                if len(sents) >= 2:
                    rsf_stats["clean_bases_multi"] += 1
                    base_multi.append(entry)
                else:
                    rsf_stats["clean_bases_single"] += 1
                    base_single_sci.append(entry)
    audit["ruscifact"] = rsf_stats

    # 5. SberQuAD (donor corpus ONLY — 9,000+ disjoint Russian encyclopedic contexts)
    sq_path = dataset_dir / "6_sberquad" / "sberquad_train.parquet"
    sq_stats = {"present": sq_path.exists(), "raw_unique_contexts": 0, "rejected_overlap": 0, "donor_docs": 0}
    seen_sq_docs: set[str] = set()
    if sq_path.exists() and pq is not None:
        for row in pq.read_table(sq_path).to_pylist():
            doc = clean_whitespace(row.get("context") or "")
            if not doc:
                continue
            dh = sha256_text(doc)
            if dh in seen_sq_docs:
                continue
            seen_sq_docs.add(dh)
            sq_stats["raw_unique_contexts"] += 1
            if blacklist.overlaps_document(doc):
                sq_stats["rejected_overlap"] += 1
                continue
            if cyr_ratio(doc) >= 0.85:
                sq_stats["donor_docs"] += 1
                harvest_donors("sberquad", doc)
    audit["sberquad"] = sq_stats

    audit["absent_local_corpora"] = [
        name
        for name in ("DFKonGH/ru_hallucination_detector", "Vikhr")
        if not (dataset_dir / name).exists()
    ]

    return {
        "base_multi": base_multi,
        "base_single_sci": base_single_sci,
        "donors_short": dict(donors_short),
        "donors_clause": dict(donors_clause),
        "donors_whole": dict(donors_whole),
        "audit": audit,
    }


# ---------------------------------------------------------------------------
# Non-confirmation control & cross-corpus donor selection
# ---------------------------------------------------------------------------

PREFERRED_DONORS = {
    "ragtruth_ru": ("sberquad", "rufacts", "mfava_ru", "ruscifact"),
    "rufacts": ("sberquad", "ragtruth_ru", "mfava_ru", "ruscifact"),
    "mfava_ru": ("sberquad", "ruscifact", "rufacts", "ragtruth_ru"),
    # Scientific abstracts stay strictly in encyclopedic/scientific Russian register
    "ruscifact": ("sberquad", "mfava_ru"),
}


def check_unconfirmed(
    clause: str,
    document: str,
    base_answer: str,
    doc_stems: set[str],
    doc_grams: set[str],
    doc_nums: set[str],
    doc_propers: set[str],
    is_short: bool = False,
) -> tuple[bool, dict]:
    """Verify clause is strictly unconfirmed by document and distinct from base answer."""
    cstems = content_stems(clause)
    if not cstems:
        return False, {"reason": "no_content_stems"}
    shared_stems = cstems & doc_stems
    stem_ov = len(shared_stems) / len(cstems)
    max_shared_stems = 0 if is_short else 2
    max_stem_ov = 0.0 if is_short else 0.15
    if len(shared_stems) > max_shared_stems or stem_ov > max_stem_ov:
        return False, {"reason": "stem_overlap", "stem_ov": stem_ov}

    cgrams = ngram_set(clause)
    ngram_ov = len(cgrams & doc_grams) / len(cgrams) if cgrams else 1.0
    if ngram_ov > 0.25:
        return False, {"reason": "ngram_overlap", "ngram_ov": ngram_ov}

    cnums = set(NUMBER_TOKEN_RE.findall(clause))
    if cnums and (cnums & doc_nums):
        return False, {"reason": "shared_number"}

    cpropers = {w.lower() for w in PROPER_NOUN_RE.findall(clause)}
    if cpropers and (cpropers & doc_propers):
        return False, {"reason": "shared_proper_noun"}

    if max_contiguous_word_overlap(clause, document) > 2:
        return False, {"reason": "contiguous_doc_overlap"}

    if base_answer and max_contiguous_word_overlap(clause, base_answer) > 2:
        return False, {"reason": "contiguous_ans_overlap"}

    return True, {
        "stem_overlap": round(stem_ov, 4),
        "ngram_overlap": round(ngram_ov, 4),
        "shared_stems": len(shared_stems),
    }


def compute_risk_and_confidence(stem_ov: float, ngram_ov: float, style: str) -> tuple[float, float]:
    """Deterministic risk and confidence scores derived from non-support metrics."""
    style_bonus = {"whole": 0.02, "clause": 0.01, "short": 0.0}[style]
    risk = 1.0 - 0.60 * stem_ov - 0.30 * ngram_ov + style_bonus
    conf = 1.0 - 0.45 * stem_ov - 0.20 * ngram_ov + style_bonus
    return (
        round(min(0.99, max(0.75, risk)), 4),
        round(min(0.99, max(0.80, conf)), 4),
    )


def classify_span_style(start: int, end: int, answer_len: int) -> str:
    span_len = end - start + 1
    rel_len = span_len / max(1, answer_len)
    if rel_len >= 0.85:
        return "whole"
    if span_len < 40:
        return "short"
    return "clause"


class DonorIndex:
    """Fast cross-corpus donor sampler with global zero-reuse tracking."""

    def __init__(
        self,
        pools_by_corpus: dict[str, list[str]],
        shared_used: set[str],
        rng: random.Random,
        enable_stem_index: bool = True,
    ) -> None:
        self.pools: dict[str, list[str]] = {}
        self.cursors: dict[str, int] = {}
        self.used = shared_used
        self.by_stem: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for corpus, items in sorted(pools_by_corpus.items()):
            shuffled = list(items)
            rng.shuffle(shuffled)
            self.pools[corpus] = shuffled
            self.cursors[corpus] = 0
            if enable_stem_index:
                for sent in shuffled:
                    for st in content_stems(sent):
                        if len(st) >= 5:
                            self.by_stem[corpus][st].append(sent)

    def _is_used(self, cand: str, exclude: set[str]) -> bool:
        if cand in self.used or cand in exclude:
            return True
        parts = split_ru_sentences(cand)
        if len(parts) > 1 and any(p in self.used or p in exclude for p in parts):
            return True
        return False

    def _mark_used(self, cand: str) -> None:
        self.used.add(cand)
        for p in split_ru_sentences(cand):
            self.used.add(p)

    def pick(
        self,
        base_corpus: str,
        document: str,
        base_answer: str,
        doc_stems: set[str],
        doc_grams: set[str],
        doc_nums: set[str],
        doc_propers: set[str],
        rng: random.Random,
        is_short: bool = False,
        prefer_thematic: bool = False,
        exclude_texts: set[str] | None = None,
    ) -> tuple[str, str, dict] | None:
        exclude = exclude_texts or set()
        donor_order = list(PREFERRED_DONORS.get(base_corpus, tuple(sorted(self.pools.keys()))))
        donor_order = [c for c in donor_order if c != base_corpus and c in self.pools and self.pools[c]]
        if not donor_order:
            return None

        if len(donor_order) > 1 and rng.random() < 0.45:
            idx = rng.randrange(len(donor_order))
            donor_order = donor_order[idx:] + donor_order[:idx]

        if prefer_thematic and not is_short:
            broad_stems = [st for st in sorted(doc_stems) if len(st) >= 5]
            rng.shuffle(broad_stems)
            for corpus in donor_order:
                stem_map = self.by_stem.get(corpus)
                if not stem_map:
                    continue
                for st in broad_stems[:12]:
                    cands = stem_map.get(st)
                    if not cands:
                        continue
                    for cand in cands[:8]:
                        if self._is_used(cand, exclude):
                            continue
                        ok, meta = check_unconfirmed(
                            cand, document, base_answer, doc_stems, doc_grams, doc_nums, doc_propers, is_short=False
                        )
                        if ok:
                            self._mark_used(cand)
                            return cand, corpus, dict(meta, thematic_bridge=st)

        for corpus in donor_order:
            pool = self.pools[corpus]
            n = len(pool)
            cursor = self.cursors[corpus]
            checked = 0
            while checked < min(n, 350):
                cand = pool[(cursor + checked) % n]
                checked += 1
                if self._is_used(cand, exclude):
                    continue
                ok, meta = check_unconfirmed(
                    cand, document, base_answer, doc_stems, doc_grams, doc_nums, doc_propers, is_short=is_short
                )
                if ok:
                    self.cursors[corpus] = (cursor + checked) % n
                    self._mark_used(cand)
                    return cand, corpus, meta
        return None


# ---------------------------------------------------------------------------
# Pair assembly with exact span coordinate tracking
# ---------------------------------------------------------------------------

def insert_clauses_into_sents(
    base_sents: list[str],
    insertions: list[tuple[int, str, str, str, dict]],
) -> tuple[str, list[dict]]:
    by_slot: dict[int, list[tuple[str, str, str, dict]]] = defaultdict(list)
    for slot, text, dcorpus, pos_lbl, meta in insertions:
        by_slot[slot].append((text, dcorpus, pos_lbl, meta))

    pieces: list[tuple[str, bool, str, str, dict | None]] = []
    for slot in range(len(base_sents) + 1):
        for text, dcorpus, pos_lbl, meta in by_slot.get(slot, ()):
            pieces.append((text, True, dcorpus, pos_lbl, meta))
        if slot < len(base_sents):
            pieces.append((base_sents[slot], False, "", "", None))

    out_parts: list[str] = []
    raw_spans: list[dict] = []
    cursor = 0
    for idx, (text, is_hallu, dcorpus, pos_lbl, meta) in enumerate(pieces):
        if idx > 0:
            out_parts.append(" ")
            cursor += 1
        start = cursor
        end = cursor + len(text) - 1
        out_parts.append(text)
        cursor += len(text)
        if is_hallu:
            raw_spans.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "donor_corpus": dcorpus,
                    "position": pos_lbl,
                    "meta": meta or {},
                }
            )
    answer = "".join(out_parts)
    ans_len = len(answer)
    final_spans: list[dict] = []
    for sp in raw_spans:
        assert answer[sp["start"] : sp["end"] + 1] == sp["text"], "Exact span slice invariant failed"
        style = classify_span_style(sp["start"], sp["end"], ans_len)
        stem_ov = sp["meta"].get("stem_overlap", 0.0)
        ngram_ov = sp["meta"].get("ngram_overlap", 0.0)
        risk, conf = compute_risk_and_confidence(stem_ov, ngram_ov, style)
        final_spans.append(
            {
                "text": sp["text"],
                "start": sp["start"],
                "end": sp["end"],
                "type": classify(sp["text"], "Unconfirmed"),
                "type_src": "Unconfirmed",
                "style": style,
                "position": sp["position"],
                "risk": risk,
                "confidence": conf,
                "donor_corpus": sp["donor_corpus"],
                "doc_stem_overlap": stem_ov,
                "doc_ngram_overlap": ngram_ov,
                "evidence_ids": [],
            }
        )
    return answer, final_spans


def choose_middle_slot(base_sents: list[str], rng: random.Random) -> int | None:
    good_slots = [
        k
        for k in range(1, len(base_sents))
        if not ANAPHORA_NEXT_RE.match(base_sents[k])
    ]
    if not good_slots:
        return None
    return rng.choice(good_slots)


def choose_two_nonadjacent_slots(base_sents: list[str], rng: random.Random) -> list[tuple[int, str]] | None:
    m = len(base_sents)
    if m < 2:
        return None
    candidates: list[list[tuple[int, str]]] = [[(0, "start"), (m, "end")]]
    for k in range(1, m):
        if not ANAPHORA_NEXT_RE.match(base_sents[k]):
            candidates.append([(0, "start"), (k, "middle")])
            candidates.append([(k, "middle"), (m, "end")])
    for k1 in range(1, m - 1):
        for k2 in range(k1 + 1, m):
            if not ANAPHORA_NEXT_RE.match(base_sents[k1]) and not ANAPHORA_NEXT_RE.match(base_sents[k2]):
                candidates.append([(k1, "middle"), (k2, "middle")])
    return rng.choice(candidates)


def make_pair_record(
    pair_idx: int,
    base: dict,
    model_output: str,
    spans: list[dict],
    mode: str,
    donor_corpora: list[str],
    max_ov: float,
) -> dict:
    doc = base["document_text"]
    return {
        "pair_id": f"ruclv19_{pair_idx:06d}",
        "source": "ru_clause_gen_v19",
        "base_corpus": base["base_corpus"],
        "donor_corpora": sorted(set(donor_corpora)),
        "domain": base["domain"],
        "doc_id": sha256_text(doc)[:16],
        "doc_ids": spanver_doc_ids(doc),
        "document_text": doc,
        "model_output": model_output,
        "spans": spans,
        "clean": len(spans) == 0,
        "label_source": "programmatic_clause_v19",
        "generation": {
            "mode": mode,
            "n_insertions": len(spans),
            "positions": [sp["position"] for sp in spans],
            "styles": [sp["style"] for sp in spans],
            "max_doc_stem_overlap": round(max_ov, 4),
            "base_id": base["base_id"],
        },
        "split": "train",
    }


def generate_dataset(corpora: dict, seed: int = 42) -> tuple[list[dict], dict]:
    """Programmatically generate the full v19 clause-hallucination dataset."""
    rng = random.Random(seed)
    base_multi = list(corpora["base_multi"])
    base_single_sci = list(corpora["base_single_sci"])
    rng.shuffle(base_multi)
    rng.shuffle(base_single_sci)

    shared_used: set[str] = set()
    short_idx = DonorIndex(corpora["donors_short"], shared_used, rng, enable_stem_index=False)
    clause_idx = DonorIndex(corpora["donors_clause"], shared_used, rng, enable_stem_index=True)
    whole_idx = DonorIndex(corpora["donors_whole"], shared_used, rng, enable_stem_index=True)

    rejected_counters: Counter[str] = Counter()
    pairs: list[dict] = []
    seen_answers: set[str] = set()

    n_multi = len(base_multi)
    n_clause_target = int(round(n_multi * 0.56))
    n_short_target = int(round(n_multi * 0.14))
    n_whole_multi_target = int(round(n_multi * 0.10))

    multi_clause_bases = base_multi[:n_clause_target]
    multi_short_bases = base_multi[n_clause_target : n_clause_target + n_short_target]
    multi_whole_bases = base_multi[n_clause_target + n_short_target : n_clause_target + n_short_target + n_whole_multi_target]
    multi_neg_bases = base_multi[n_clause_target + n_short_target + n_whole_multi_target :]

    n_sci = len(base_single_sci)
    n_sci_whole = int(round(n_sci * 0.65))
    sci_whole_bases = base_single_sci[:n_sci_whole]
    sci_neg_bases = base_single_sci[n_sci_whole:]

    def prep_doc_features(base: dict):
        doc = base["document_text"]
        return (
            content_stems(doc),
            ngram_set(doc),
            set(NUMBER_TOKEN_RE.findall(doc)),
            {w.lower() for w in PROPER_NOUN_RE.findall(doc)},
        )

    pos_cycle = ["start", "middle", "end"]

    # 1. CLAUSE multi-sentence insertions (1-insert & 2-insert non-adjacent)
    for i, base in enumerate(multi_clause_bases):
        doc_stems, doc_grams, doc_nums, doc_propers = prep_doc_features(base)
        base_sents = base["sents"]
        base_ans = " ".join(base_sents)
        is_double = (i % 5 == 0)
        insertions: list[tuple[int, str, str, str, dict]] = []
        donor_corpora: list[str] = []

        if is_double:
            slots = choose_two_nonadjacent_slots(base_sents, rng)
            if not slots:
                rejected_counters["no_double_slot"] += 1
                continue
            picked_texts: set[str] = set()
            ok_all = True
            for slot_idx, pos_lbl in slots:
                picked = clause_idx.pick(
                    base["base_corpus"],
                    base["document_text"],
                    base_ans,
                    doc_stems,
                    doc_grams,
                    doc_nums,
                    doc_propers,
                    rng,
                    is_short=False,
                    prefer_thematic=(rng.random() < 0.50),
                    exclude_texts=picked_texts,
                )
                if not picked:
                    ok_all = False
                    break
                ctext, dcorp, meta = picked
                picked_texts.add(ctext)
                donor_corpora.append(dcorp)
                insertions.append((slot_idx, ctext, dcorp, pos_lbl, meta))
            if not ok_all:
                rejected_counters["no_donor_double_clause"] += 1
                continue
            mode = "clause_insert_2"
        else:
            desired_pos = pos_cycle[i % 3]
            if desired_pos == "middle":
                mid = choose_middle_slot(base_sents, rng)
                if mid is not None:
                    slot_idx, pos_lbl = mid, "middle"
                else:
                    slot_idx, pos_lbl = (0, "start") if (i % 2 == 0) else (len(base_sents), "end")
            elif desired_pos == "start":
                slot_idx, pos_lbl = 0, "start"
            else:
                slot_idx, pos_lbl = len(base_sents), "end"

            picked = clause_idx.pick(
                base["base_corpus"],
                base["document_text"],
                base_ans,
                doc_stems,
                doc_grams,
                doc_nums,
                doc_propers,
                rng,
                is_short=False,
                prefer_thematic=(rng.random() < 0.50),
            )
            if not picked:
                rejected_counters["no_donor_clause"] += 1
                continue
            ctext, dcorp, meta = picked
            donor_corpora.append(dcorp)
            insertions.append((slot_idx, ctext, dcorp, pos_lbl, meta))
            mode = "clause_insert_1"

        ans, spans = insert_clauses_into_sents(base_sents, insertions)
        if not all(sp["style"] == "clause" for sp in spans):
            rejected_counters["style_mismatch_clause"] += 1
            continue
        ah = sha256_text(ans)
        if ah in seen_answers:
            rejected_counters["dup_answer"] += 1
            continue
        seen_answers.add(ah)
        max_ov = max(sp["doc_stem_overlap"] for sp in spans)
        pairs.append(make_pair_record(len(pairs) + 1, base, ans, spans, mode, donor_corpora, max_ov))

    # 2. SHORT multi-sentence clause insertions (20-39 char complete Russian sentence)
    for i, base in enumerate(multi_short_bases):
        doc_stems, doc_grams, doc_nums, doc_propers = prep_doc_features(base)
        base_sents = base["sents"]
        base_ans = " ".join(base_sents)
        desired_pos = pos_cycle[i % 3]
        if desired_pos == "middle":
            mid = choose_middle_slot(base_sents, rng)
            if mid is not None:
                slot_idx, pos_lbl = mid, "middle"
            else:
                slot_idx, pos_lbl = (0, "start") if (i % 2 == 0) else (len(base_sents), "end")
        elif desired_pos == "start":
            slot_idx, pos_lbl = 0, "start"
        else:
            slot_idx, pos_lbl = len(base_sents), "end"

        picked = short_idx.pick(
            base["base_corpus"],
            base["document_text"],
            base_ans,
            doc_stems,
            doc_grams,
            doc_nums,
            doc_propers,
            rng,
            is_short=True,
            prefer_thematic=False,
        )
        if not picked:
            rejected_counters["no_donor_short"] += 1
            continue
        ctext, dcorp, meta = picked
        ans, spans = insert_clauses_into_sents(base_sents, [(slot_idx, ctext, dcorp, pos_lbl, meta)])
        if not all(sp["style"] == "short" for sp in spans):
            rejected_counters["style_mismatch_short"] += 1
            continue
        ah = sha256_text(ans)
        if ah in seen_answers:
            rejected_counters["dup_answer"] += 1
            continue
        seen_answers.add(ah)
        pairs.append(
            make_pair_record(
                len(pairs) + 1, base, ans, spans, "short_clause_insert_1", [dcorp], spans[0]["doc_stem_overlap"]
            )
        )

    # 3. WHOLE clause hallucinations (1-2 contiguous grammatical sentences from another corpus)
    for base in multi_whole_bases + sci_whole_bases:
        doc_stems, doc_grams, doc_nums, doc_propers = prep_doc_features(base)
        picked = whole_idx.pick(
            base["base_corpus"],
            base["document_text"],
            "",
            doc_stems,
            doc_grams,
            doc_nums,
            doc_propers,
            rng,
            is_short=False,
            prefer_thematic=(rng.random() < 0.40),
        )
        if not picked:
            rejected_counters["no_donor_whole"] += 1
            continue
        wtext, dcorp, meta = picked
        wsents = split_ru_sentences(wtext)
        if not wsents:
            rejected_counters["bad_whole_split"] += 1
            continue
        all_unconf = True
        for ws in wsents:
            ok_s, _ = check_unconfirmed(
                ws, base["document_text"], "", doc_stems, doc_grams, doc_nums, doc_propers, is_short=False
            )
            if not ok_s:
                all_unconf = False
                break
        if not all_unconf:
            rejected_counters["whole_subsent_overlap"] += 1
            continue
        ah = sha256_text(wtext)
        if ah in seen_answers:
            rejected_counters["dup_answer"] += 1
            continue
        seen_answers.add(ah)
        stem_ov = meta.get("stem_overlap", 0.0)
        ngram_ov = meta.get("ngram_overlap", 0.0)
        risk, conf = compute_risk_and_confidence(stem_ov, ngram_ov, "whole")
        span = {
            "text": wtext,
            "start": 0,
            "end": len(wtext) - 1,
            "type": classify(wtext, "Unconfirmed"),
            "type_src": "Unconfirmed",
            "style": "whole",
            "position": "whole",
            "risk": risk,
            "confidence": conf,
            "donor_corpus": dcorp,
            "doc_stem_overlap": stem_ov,
            "doc_ngram_overlap": ngram_ov,
            "evidence_ids": [],
        }
        pairs.append(
            make_pair_record(len(pairs) + 1, base, wtext, [span], "whole_clause_cross_corpus", [dcorp], stem_ov)
        )

    # 4. CLEAN NEGATIVE pairs (strictly disjoint unique documents, spans == [])
    for base in multi_neg_bases + sci_neg_bases:
        ans = " ".join(base["sents"])
        if base["min_support"] < 0.30 or base["mean_support"] < 0.50:
            rejected_counters["weak_negative_support"] += 1
            continue
        ah = sha256_text(ans)
        if ah in seen_answers:
            rejected_counters["dup_answer"] += 1
            continue
        seen_answers.add(ah)
        pairs.append(make_pair_record(len(pairs) + 1, base, ans, [], "clean_negative", [], 0.0))

    rng.shuffle(pairs)
    for idx, pair in enumerate(pairs, start=1):
        pair["pair_id"] = f"ruclv19_{idx:06d}"

    return pairs, dict(rejected_counters)


# ---------------------------------------------------------------------------
# Verification & Real-Style Diagnostics (ШАГ 3 & ШАГ 5)
# ---------------------------------------------------------------------------

def oracle_cand_top1(pair: dict) -> list[dict]:
    gold = pair.get("spans") or []
    if not gold:
        return []
    cands = candidate_spans(pair.get("model_output") or "")
    best = None
    best_v = 0.0
    for cs, ce, _ in cands:
        for g in gold:
            v = iou(cs, ce, g["start"], g["end"])
            if v > best_v:
                best_v = v
                best = {"start": cs, "end": ce, "type": g.get("type", "unsupported_addition")}
    return [best] if best and best_v >= IOU_THRESHOLD else []


def oracle_cand_multi(pair: dict) -> list[dict]:
    gold = pair.get("spans") or []
    if not gold:
        return []
    cands = candidate_spans(pair.get("model_output") or "")
    out: list[dict] = []
    for g in gold:
        best = None
        best_v = 0.0
        for cs, ce, _ in cands:
            v = iou(cs, ce, g["start"], g["end"])
            if v > best_v:
                best_v = v
                best = {"start": cs, "end": ce, "type": g.get("type", "unsupported_addition")}
        if best and best_v >= IOU_THRESHOLD:
            out.append(best)
    return out


def oracle_sent_multi(pair: dict) -> list[dict]:
    gold = pair.get("spans") or []
    if not gold:
        return []
    sents = sentence_spans(pair.get("model_output") or "")
    out: list[dict] = []
    used: set[int] = set()
    for g in gold:
        best = None
        best_v = 0.0
        for idx, (ss, se) in enumerate(sents):
            if idx in used:
                continue
            v = iou(ss, se, g["start"], g["end"])
            if v > best_v:
                best_v = v
                best = (idx, ss, se)
        if best and best_v >= IOU_THRESHOLD:
            used.add(best[0])
            out.append({"start": best[1], "end": best[2], "type": g.get("type", "unsupported_addition")})
    return out


def oracle_token_snap_expand(pair: dict) -> list[dict]:
    gold = pair.get("spans") or []
    if not gold:
        return []
    ans = pair.get("model_output") or ""
    pos = token_positions(ans)
    cur = None
    spans: list[dict] = []
    for ts, te in pos:
        hit = any(min(te, g["end"]) - max(ts, g["start"]) >= 0 for g in gold)
        if hit:
            if cur is None:
                cur = {"start": ts, "end": te, "type": "unsupported_addition"}
            else:
                cur["end"] = te
        elif cur is not None:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    return snapped(spans, ans, "expand")


def summarize_style_and_oracles(pairs: list[dict], run_lexical_baseline: bool = True) -> dict:
    pos_pairs = [p for p in pairs if p.get("spans")]
    neg_pairs = [p for p in pairs if not p.get("spans")]
    all_spans = [sp for p in pairs for sp in (p.get("spans") or [])]

    span_lens = [sp["end"] - sp["start"] + 1 for sp in all_spans]
    rel_lens = [
        (sp["end"] - sp["start"] + 1) / max(1, len(p.get("model_output") or ""))
        for p in pairs
        for sp in (p.get("spans") or [])
    ]
    ans_lens_pos = [len(p.get("model_output") or "") for p in pos_pairs]
    ans_lens_neg = [len(p.get("model_output") or "") for p in neg_pairs]
    sent_counts_pos = [len(sentence_spans(p.get("model_output") or "")) for p in pos_pairs]

    style_counts = Counter({"whole": 0, "clause": 0, "short": 0})
    type_counts = Counter({k: 0 for k in TYPE_KEYS})
    pos_counts: Counter[str] = Counter()
    spans_per_pos: Counter[int] = Counter()
    for p in pos_pairs:
        al = len(p.get("model_output") or "")
        spans_per_pos[len(p["spans"])] += 1
        for sp in p["spans"]:
            st = sp.get("style") or classify_span_style(sp["start"], sp["end"], al)
            style_counts[st] += 1
            type_counts[sp.get("type", "unsupported_addition")] += 1
            if "position" in sp:
                pos_counts[sp["position"]] += 1

    total_spans = len(all_spans)
    sorted_lens = sorted(span_lens)

    def quant(arr: list[int], q: float) -> float:
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, int(round(q * (len(arr) - 1))))
        return float(arr[idx])

    m_top1 = metrics(pairs, oracle_cand_top1)
    m_multi = metrics(pairs, oracle_cand_multi)
    m_sent = metrics(pairs, oracle_sent_multi)
    m_snap = metrics(pairs, oracle_token_snap_expand)

    result = {
        "n_pairs": len(pairs),
        "n_positive_pairs": len(pos_pairs),
        "n_negative_pairs": len(neg_pairs),
        "positive_ratio": round(len(pos_pairs) / max(1, len(pairs)), 4),
        "n_spans": total_spans,
        "spans_per_positive_pair": {str(k): v for k, v in sorted(spans_per_pos.items())},
        "style_distribution": {
            k: {
                "count": style_counts[k],
                "share": round(style_counts[k] / max(1, total_spans), 4),
            }
            for k in ("whole", "clause", "short")
        },
        "contract_type_distribution": dict(sorted(type_counts.items())),
        "position_distribution": dict(sorted(pos_counts.items())),
        "span_length_chars": {
            "mean": round(statistics.mean(span_lens), 2) if span_lens else 0.0,
            "median": round(statistics.median(span_lens), 2) if span_lens else 0.0,
            "p25": quant(sorted_lens, 0.25),
            "p75": quant(sorted_lens, 0.75),
            "min": min(span_lens) if span_lens else 0,
            "max": max(span_lens) if span_lens else 0,
        },
        "span_relative_length": {
            "mean": round(statistics.mean(rel_lens), 4) if rel_lens else 0.0,
            "median": round(statistics.median(rel_lens), 4) if rel_lens else 0.0,
        },
        "answer_length_chars": {
            "pos_mean": round(statistics.mean(ans_lens_pos), 2) if ans_lens_pos else 0.0,
            "pos_median": round(statistics.median(ans_lens_pos), 2) if ans_lens_pos else 0.0,
            "neg_mean": round(statistics.mean(ans_lens_neg), 2) if ans_lens_neg else 0.0,
            "neg_median": round(statistics.median(ans_lens_neg), 2) if ans_lens_neg else 0.0,
            "pos_sents_mean": round(statistics.mean(sent_counts_pos), 2) if sent_counts_pos else 0.0,
            "pos_sents_median": round(statistics.median(sent_counts_pos), 2) if sent_counts_pos else 0.0,
        },
        "oracle_ceiling": {
            "candidate_ceiling": eval_ceiling(pairs),
            "oracle_candidate_top1": {
                "span_f1": m_top1["span_level"]["f1"],
                "span_precision": m_top1["span_level"]["precision"],
                "span_recall": m_top1["span_level"]["recall"],
                "token_f1": m_top1["token_level"]["f1"],
            },
            "oracle_candidate_multi": {
                "span_f1": m_multi["span_level"]["f1"],
                "span_precision": m_multi["span_level"]["precision"],
                "span_recall": m_multi["span_level"]["recall"],
                "token_f1": m_multi["token_level"]["f1"],
            },
            "oracle_sentence_multi": {
                "span_f1": m_sent["span_level"]["f1"],
                "span_precision": m_sent["span_level"]["precision"],
                "span_recall": m_sent["span_level"]["recall"],
                "token_f1": m_sent["token_level"]["f1"],
            },
            "oracle_token_snap_expand": {
                "span_f1": m_snap["span_level"]["f1"],
                "span_precision": m_snap["span_level"]["precision"],
                "span_recall": m_snap["span_level"]["recall"],
                "token_f1": m_snap["token_level"]["f1"],
            },
        },
    }
    if run_lexical_baseline:
        lex = evaluate_pairs(pairs)
        result["zero_shot_lexical_verify"] = {
            "span_f1": lex["span_level"]["f1"],
            "span_precision": lex["span_level"]["precision"],
            "span_recall": lex["span_level"]["recall"],
            "token_f1": lex["token_level"]["f1"],
            "pair_f1": lex["pair_level"]["f1"],
            "fpr": lex["fpr"],
            "type_accuracy": lex["type_accuracy"],
        }
    return result


def verify_zero_overlap(pairs: list[dict], blacklist: RusHalluBlacklist) -> dict:
    """Independently verify doc_overlap == 0 against RusHallu-RAG across all granularity levels."""
    doc_id_hits = 0
    full_hash_hits = 0
    subdoc_hash_hits = 0
    prefix_hits = 0
    ngram8_hits = 0
    clause_hits = 0

    for p in pairs:
        doc = p.get("document_text") or ""
        nt = norm_text(doc)
        h = hashlib.sha256(nt.encode("utf-8")).hexdigest()
        if any(did in blacklist.doc_id_set for did in (p.get("doc_ids") or [])):
            doc_id_hits += 1
        if h in blacklist.full_hashes:
            full_hash_hits += 1
        if h in blacklist.subdoc_hashes:
            subdoc_hash_hits += 1
        if len(nt) >= 30 and nt[:100] in blacklist.prefixes:
            prefix_hits += 1
        ng = word_ngrams(doc, 8)
        if ng and (ng & blacklist.ngrams8):
            ngram8_hits += 1
        for sp in p.get("spans") or []:
            if blacklist.overlaps_clause(sp.get("text") or ""):
                clause_hits += 1

    total = doc_id_hits + full_hash_hits + subdoc_hash_hits + prefix_hits + ngram8_hits + clause_hits
    return {
        "doc_overlap": total,
        "by_doc_id": doc_id_hits,
        "by_full_doc_sha256": full_hash_hits,
        "by_subdoc_sha256": subdoc_hash_hits,
        "by_prefix_100": prefix_hits,
        "by_word_8gram": ngram8_hits,
        "by_inserted_clause": clause_hits,
        "rushallu_gold_docs_checked": blacklist.gold_doc_count,
        "rushallu_unlabeled_docs_checked": blacklist.unlabeled_doc_count,
    }


def verify_internal_integrity(pairs: list[dict]) -> dict:
    doc_hashes = [sha256_text(p["document_text"]) for p in pairs]
    ans_hashes = [sha256_text(p["model_output"]) for p in pairs]
    donor_spans = [sp["text"] for p in pairs for sp in p["spans"]]
    cross_corpus_violations = sum(
        1
        for p in pairs
        for sp in p["spans"]
        if sp.get("donor_corpus") == p.get("base_corpus")
    )
    high_overlap_spans = sum(
        1
        for p in pairs
        for sp in p["spans"]
        if sp.get("doc_stem_overlap", 0.0) > 0.15 or sp.get("doc_ngram_overlap", 0.0) > 0.25
    )
    missing_fields = sum(
        1
        for p in pairs
        for sp in p["spans"]
        if not all(k in sp for k in ("start", "end", "risk", "type", "confidence", "text", "style"))
    )
    return {
        "unique_documents": len(set(doc_hashes)),
        "duplicate_documents": len(doc_hashes) - len(set(doc_hashes)),
        "unique_answers": len(set(ans_hashes)),
        "duplicate_answers": len(ans_hashes) - len(set(ans_hashes)),
        "unique_donor_spans": len(set(donor_spans)),
        "duplicate_donor_spans": len(donor_spans) - len(set(donor_spans)),
        "cross_corpus_violations": cross_corpus_violations,
        "high_overlap_spans": high_overlap_spans,
        "missing_required_span_fields": missing_fields,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Programmatic Russian clause hallucination generator (v19)")
    parser.add_argument("--dataset", default="Dataset", help="Path to local Dataset directory")
    parser.add_argument("--out-jsonl", default="artifacts/ru_clause_gen_v19.jsonl", help="Output JSONL path")
    parser.add_argument("--out-report", default="artifacts/ru_clause_gen_v19.json", help="Output JSON report path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    out_jsonl = Path(args.out_jsonl)
    out_report = Path(args.out_report)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    blacklist = RusHalluBlacklist(dataset_dir)
    corpora = load_local_corpora(dataset_dir, blacklist)
    pairs, rejected = generate_dataset(corpora, seed=args.seed)

    with open(out_jsonl, "w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    qc = qc_inspect(pairs)
    overlap = verify_zero_overlap(pairs, blacklist)
    integrity = verify_internal_integrity(pairs)
    synth_stats = summarize_style_and_oracles(pairs, run_lexical_baseline=True)
    gold_ref_stats = summarize_style_and_oracles(blacklist.gold_pairs, run_lexical_baseline=True)

    base_corpus_counts = Counter(p["base_corpus"] for p in pairs)
    donor_corpus_counts = Counter(sp["donor_corpus"] for p in pairs for sp in p["spans"])
    mode_counts = Counter(p["generation"]["mode"] for p in pairs)
    cross_matrix = Counter(
        f"{p['base_corpus']} <- {sp['donor_corpus']}" for p in pairs for sp in p["spans"]
    )

    dataset_sha256 = sha256_file(out_jsonl)
    script_path = Path(__file__).resolve()
    try:
        rel_script = str(script_path.relative_to(ROOT))
    except ValueError:
        rel_script = str(script_path)

    report = {
        "version": "ru_clause_gen_v19",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "programmatic_only": True,
        "manual_annotation": False,
        "flip_augmentation_used": False,
        "generator_script": rel_script,
        "generator_script_sha256": sha256_file(script_path),
        "dataset_path": str(out_jsonl),
        "dataset_bytes": out_jsonl.stat().st_size,
        "dataset_sha256": dataset_sha256,
        "doc_overlap": overlap["doc_overlap"],
        "doc_overlap_details": overlap,
        "qc": qc,
        "integrity": integrity,
        "sources": {
            "local_corpora_used": sorted(
                k for k, v in corpora["audit"].items() if isinstance(v, dict) and v.get("present")
            ),
            "absent_local_corpora": corpora["audit"].get("absent_local_corpora", []),
            "rushallu_rag_excluded": [
                "Dataset/1_rushallu_rag_gold/sberquad-rag.csv",
                "Dataset/1_rushallu_rag_gold/ruscibench-rag.csv",
                "Dataset/2_rushallu_rag_pairs/sberquad-train-rag.parquet",
            ],
            "corpus_audit": {
                k: v for k, v in corpora["audit"].items() if k != "absent_local_corpora"
            },
            "donor_pool_sizes": {
                "short": {k: len(v) for k, v in sorted(corpora["donors_short"].items())},
                "clause": {k: len(v) for k, v in sorted(corpora["donors_clause"].items())},
                "whole": {k: len(v) for k, v in sorted(corpora["donors_whole"].items())},
            },
            "base_corpus_pair_counts": dict(sorted(base_corpus_counts.items())),
            "donor_corpus_span_counts": dict(sorted(donor_corpus_counts.items())),
            "cross_corpus_span_matrix": dict(sorted(cross_matrix.items(), key=lambda kv: (-kv[1], kv[0]))),
            "generation_mode_counts": dict(sorted(mode_counts.items())),
            "rejected_during_generation": dict(sorted(rejected.items())),
        },
        "stats": synth_stats,
        "reference_rushallu_rag_gold": gold_ref_stats,
        "gates": {
            "qc_ok": bool(qc["ok"]),
            "zero_control_chars": qc["control_pairs"] == 0,
            "doc_overlap_zero": overlap["doc_overlap"] == 0,
            "cross_corpus_strict": integrity["cross_corpus_violations"] == 0,
            "zero_duplicate_docs": integrity["duplicate_documents"] == 0,
            "zero_duplicate_donors": integrity["duplicate_donor_spans"] == 0,
            "zero_high_overlap_spans": integrity["high_overlap_spans"] == 0,
            "zero_missing_span_fields": integrity["missing_required_span_fields"] == 0,
            "oracle_cand_top1_f1_ge_0_60": synth_stats["oracle_ceiling"]["oracle_candidate_top1"]["span_f1"] >= 0.60,
            "oracle_cand_multi_f1_ge_0_60": synth_stats["oracle_ceiling"]["oracle_candidate_multi"]["span_f1"] >= 0.60,
            "oracle_sent_multi_f1_ge_0_60": synth_stats["oracle_ceiling"]["oracle_sentence_multi"]["span_f1"] >= 0.60,
            "oracle_snap_f1_ge_0_60": synth_stats["oracle_ceiling"]["oracle_token_snap_expand"]["span_f1"] >= 0.60,
        },
    }
    report["gates"]["all_passed"] = all(report["gates"].values())

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset written: {out_jsonl} ({len(pairs)} pairs, {synth_stats['n_spans']} spans)")
    print(f"SHA-256: {dataset_sha256}")
    print(f"doc_overlap: {overlap['doc_overlap']} | qc_ok: {qc['ok']} | all_gates: {report['gates']['all_passed']}")
    print(f"Styles: {json.dumps(synth_stats['style_distribution'], ensure_ascii=False)}")
    print(f"Span len: {json.dumps(synth_stats['span_length_chars'], ensure_ascii=False)}")
    print(f"Oracle F1 (top1 / multi / sent / snap): "
          f"{synth_stats['oracle_ceiling']['oracle_candidate_top1']['span_f1']} / "
          f"{synth_stats['oracle_ceiling']['oracle_candidate_multi']['span_f1']} / "
          f"{synth_stats['oracle_ceiling']['oracle_sentence_multi']['span_f1']} / "
          f"{synth_stats['oracle_ceiling']['oracle_token_snap_expand']['span_f1']}")
    print(f"Report written: {out_report}")
    return 0 if report["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
