"""ПРОМПТ 63: модель посегментного отбора кандидатов.

Отличие от схемы 52 (и причина, по которой линия не закрыта настройкой):

* вход содержит ВОПРОС, и он обязателен: `[CLS] question [SEP] чанк [SEP] кандидат [SEP]`;
* предсказывается не вероятность субтокена, а решение по КАНДИДАТУ — фрагменту
  ответа (предложение, клауза, объединение соседних единиц, весь ответ);
* две головы: бинарная «есть галлюцинация» и вспомогательная на шесть типов;
* в loss добавлен soft-IoU по границам с весом 0,3: именно то, что меряет
  span-метрика, а в схеме 52 оно не оптимизировалось вовсе.

Признаки на кандидата (п. 2.4, считаются на лету, без обучения):

    max / mean / std вероятности субтокенов внутри кандидата   — прежний сигнал
    доля содержательных слов вопроса в кандидате               — релевантность
    лексическое перекрытие с документом                        — опора
    есть ли числа, даты, имена                                 — тип
    длина кандидата, позиция в ответе                          — структура
    число кандидатов в паре                                    — контекст

Вероятности субтокенов берутся у уже обученной модели поставки (она в проекте
есть и зафиксирована), поэтому «прежний сигнал» — не новая голова, а готовый
признак. Это позволяет держать размер пакета в норме п. 3.8.
"""
import math
import re

import torch
from torch import nn

from spanver.module import CAPITAL_RE, DIGIT_TOKEN_RE, document_token_set, stem

FEATURE_COUNT = 10

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
DATE_RE = re.compile(r"\b\d{1,4}[./-]\d{1,2}([./-]\d{1,4})?\b|\b(19|20)\d{2}\b")
STOP = frozenset(("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между
""").split())


def content_words(text):
    words = []
    for match in WORD_RE.finditer(text or ""):
        token = match.group(0).lower()
        if len(token) < 3 or token in STOP:
            continue
        words.append(stem(token))
    return words


def has_number(text):
    return 1.0 if DIGIT_TOKEN_RE.search(text or "") else 0.0


def has_date(text):
    return 1.0 if DATE_RE.search(text or "") else 0.0


def has_name(text):
    """Имя собственное: слово с заглавной не в начале кандидата."""
    for match in re.finditer(r"[^\W\d_]+", text or ""):
        if match.start() > 0 and CAPITAL_RE.match(match.group(0)):
            return 1.0
    return 0.0


def candidate_features(answer, candidate, question, document, question_words, document_tokens,
                       probabilities, ranks):
    """Признаки одного кандидата. `probabilities` — по субтокенам, `ranks` — их позиции."""
    left, right = candidate["start"], candidate["end"]
    text = answer[left:right + 1]
    inside = [value for (start, end), value in probabilities if min(right, end) - max(left, start) >= 0]
    if inside:
        maximum = max(inside)
        mean = sum(inside) / len(inside)
        std = math.sqrt(sum((x - mean) ** 2 for x in inside) / len(inside))
    else:
        maximum = mean = std = 0.0
    words = set(content_words(text))
    share = (sum(1 for word in question_words if word in words) / len(question_words)
             if question_words else 0.0)
    if document_tokens:
        own = content_words(text)
        overlap = (sum(1 for word in own if word in document_tokens) / len(own)) if own else 0.0
    else:
        overlap = 0.0
    return [
        maximum,
        mean,
        std,
        share,
        overlap,
        has_number(text),
        has_date(text),
        has_name(text),
        min(1.0, len(text) / 400.0),
        ranks.get((left, right), 0.0),
    ]


class CandidateModel(nn.Module):
    """rubert-tiny2 + вектор признаков кандидата, две головы.

    Энкодер тот же, что в схеме 52 (`models/rubert-tiny2`), по двум причинам из
    задания: скорость (п. 3.7) и размер пакета (п. 3.8). Кодируется пара
    «контекст + кандидат», а не весь ответ, поэтому длина входа ограничена.
    """

    def __init__(self, base, feature_count=FEATURE_COUNT, types=6):
        super().__init__()
        self.base = base
        self.feature_count = feature_count
        self.proj = nn.Linear(base.config.hidden_size + feature_count, base.config.hidden_size)
        hidden = base.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.binary = nn.Linear(hidden, 1)
        self.kind = nn.Linear(hidden, types)

    def forward(self, input_ids, attention_mask, features):
        """Признаки кандидата — один вектор на строку.

        Они относятся к кандидату целиком, а не к токену, поэтому кладутся в
        позицию [CLS]: остальные позиции получают нули. Так энкодер видит
        «контекст + кандидат» и сводку признаков в одной точке, и размер входа
        не растёт.
        """
        embeddings = self.base.bert.embeddings(input_ids)
        batch, length, hidden = embeddings.shape
        token_features = torch.zeros(batch, length, self.feature_count,
                                     dtype=embeddings.dtype, device=embeddings.device)
        token_features[:, 0, :] = features.to(embeddings.dtype)
        mixed = self.proj(torch.cat([embeddings, token_features], dim=-1))
        mask = (1.0 - attention_mask[:, None, None, :].to(mixed.dtype)) * torch.finfo(mixed.dtype).min
        hidden = self.base.bert.encoder(mixed, attention_mask=mask)[0]
        # [CLS] агрегирует пару «контекст + кандидат»
        pooled = self.dropout(hidden[:, 0])
        return self.binary(pooled).squeeze(-1), self.kind(pooled)


def soft_iou_loss(logits, targets, weights=None):
    """Soft-IoU по решению кандидата.

    Прямое обучение тому, что меряет span-метрика: пересечение выбранного
    множества кандидатов с золотым. Считается по вероятностям, поэтому
    дифференцируемо; `weights` — длина кандидата в знаках, чтобы IoU считался
    по тексту, а не по числу кандидатов.
    """
    probabilities = torch.sigmoid(logits)
    if weights is None:
        weights = torch.ones_like(probabilities)
    intersection = (probabilities * targets * weights).sum()
    union = ((probabilities + targets - probabilities * targets) * weights).sum()
    return 1.0 - intersection / union.clamp(min=1e-6)


def build_weights(model, base):
    """Заморозка энкодера, как в схеме 52 вариант v2: обучаются головы и проекция."""
    parameters = list(model.proj.parameters()) + list(model.binary.parameters()) + list(model.kind.parameters())
    embeddings = [parameter for parameter in base.bert.embeddings.parameters()]
    return parameters, embeddings
