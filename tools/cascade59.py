"""ПРОМПТ 59C: двухступенчатый отбор — порог по документной опоре, вопрос ранжирует.

Зачем. В прогоне B вопрос был подан в тот же скалярный скор, который задаёт
порог. Порог поехал вниз, FPR вырос с 0,3787 до 0,4076 и вышел за норму, а
прирост по фрагментам (+10,2 % exact span, +9,4 % pair F1) оказался спрятан за
этим. Гипотеза не провалилась — испорчена точка работы.

Что здесь. Две функции разделены:

* **порог** решает прежний документный сигнал (вероятность модели), порог
  выбирается по внутренним фолдам как в A и B — FPR должен остаться уровня A;
* **вопрос** только ранжирует кандидатов, прошедших порог, и никогда не
  опускает порог.

Каскад (зафиксирован в `artifacts/protocol59c.md`):

  1. субтокены с вероятностью >= порога фолда;
  2. если их больше одного — выбрать один по релевантности вопросу;
  3. если их ноль — нижняя полка: субтокены с вероятностью >= 0,30, выбрать
     один по релевантности.

Ступень 3 не понижает основной порог: значения ниже порога фолда не
конкурируют с прошедшими, они включаются только там, где прежняя схема не
давала ничего. Порог полки 0,30 и веса 0,5/0,5 зафиксированы до прогона.

Релевантность — из уже посчитанных векторов энкодера, без нового обучения.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.module import stem

# зафиксировано протоколом C до прогона, подбор по результату запрещён
SHELF_THRESHOLD = 0.30
WEIGHT_COS = 0.5
WEIGHT_SHARE = 0.5

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
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
    """Знаменательные слова: буквенные, не в стоп-листе, длиннее двух знаков."""
    words = []
    for match in WORD_RE.finditer(text or ""):
        token = match.group(0).lower()
        if len(token) < 3 or token in STOP:
            continue
        words.append(stem(token))
    return words


def share_of_question(question, span_text):
    """Доля знаменательных слов вопроса, встречающихся в спане."""
    wanted = content_words(question)
    if not wanted:
        return 0.0
    present = set(content_words(span_text))
    return sum(1 for word in wanted if word in present) / len(wanted)


def cosine(first, second):
    """Косинус двух усреднённых векторов (список float)."""
    if not first or not second:
        return 0.0
    dot = sum(a * b for a, b in zip(first, second))
    norm_a = sum(a * a for a in first) ** 0.5
    norm_b = sum(b * b for b in second) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def relevance(question_vector, span_vector, question, span_text):
    """0,5 * cos + 0,5 * share — формула зафиксирована в протоколе.

    Если векторов нет (отладка, тесты), косинусная часть считается нулевой, а
    доля слов вопроса — всегда: иначе выбор вырождается в «первый прошедший» и
    ранжирование не работает вовсе.
    """
    return WEIGHT_COS * cosine(question_vector, span_vector) + WEIGHT_SHARE * share_of_question(question, span_text)


def runs_of(pieces, threshold):
    """Непрерывные пробеги субтокенов с вероятностью >= threshold.

    Разрыв проверяется не только по флагу, но и по тексту: соседние piece могут
    быть разделены непокрытыми знаками, и тогда это ДВА спана, а не один. Без
    этой проверки несмежные кандидаты слипаются в один, и ранжирование выбирает
    «первый прошедший» вместо релевантного.
    """
    runs = []
    start = None
    previous_end = None
    for index, (left, right, score) in enumerate(pieces):
        passes = score >= threshold
        if passes:
            if start is None:
                start = index
            elif previous_end is not None and left > previous_end + 1:
                # между прогонами есть непокрытые знаки — закрываем текущий
                runs.append((start, index - 1))
                start = index
            previous_end = right
        elif start is not None:
            runs.append((start, index - 1))
            start = None
            previous_end = None
    if start is not None:
        runs.append((start, len(pieces) - 1))
    return runs


def run_span(pieces, run):
    """Границы прогона в знаках ответа."""
    first, last = run
    return pieces[first][0], pieces[last][1]


def select_cascade(answer, pieces, question, threshold, vectors=None, shelf=SHELF_THRESHOLD):
    """Один спан на пару: порог по опоре, выбор по релевантности.

    `vectors` — (question_vector, span_vector_by_run) или None: без векторов
    ранжирование идёт только по `share`, что тоже зафиксировано (векторы — часть
    формулы, но вырождаться в неё не могут, поэтому это режим отладки).
    """
    if not pieces:
        return None
    passed = runs_of(pieces, threshold)
    used_shelf = False
    if not passed:
        # ступень 3: только если порог не пропустил ничего
        passed = runs_of(pieces, shelf)
        used_shelf = bool(passed)
    if not passed:
        return None

    question_vector, span_vectors = (vectors or (None, {}))
    best = None
    for index, run in enumerate(passed):
        left, right = run_span(pieces, run)
        text = answer[left:right + 1]
        vector = (span_vectors or {}).get(run)
        value = relevance(question_vector, vector, question, text)
        # при равной релевантности выигрывает более ранний прогон
        if best is None or value > best[0]:
            best = (value, run, left, right)
    _, run, left, right = best
    return {
        "start": left,
        "end": right,
        "relevance": best[0],
        "runs_passed": len(passed),
        "from_shelf": used_shelf,
        "risk": max(piece[2] for piece in pieces[run[0]:run[1] + 1]),
    }
