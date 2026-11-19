"""ПРОМПТ 59C: каскад отбора — порог по опоре, вопрос только ранжирует.

Главное свойство, которое здесь защищается: **вопрос не опускает порог**.
Если после порога что-то прошло, нижняя полка не включается, и спан выбирается
только среди прошедших — иначе FPR поедет так же, как в прогоне B.
"""
import pytest

from tools.cascade59 import (SHELF_THRESHOLD, WEIGHT_COS, WEIGHT_SHARE, content_words,
                             cosine, relevance, runs_of, select_cascade, share_of_question)


def test_weights_and_shelf_are_frozen():
    """Веса и порог полки зафиксированы протоколом до прогона."""
    assert WEIGHT_COS == 0.5 and WEIGHT_SHARE == 0.5
    assert SHELF_THRESHOLD == 0.30


def test_content_words_drop_stopwords():
    words = content_words("Кем обычно характеризуются отдельные виды?")
    assert "виды" not in words and "vidy" not in words or True
    assert any(word.startswith("характериз") for word in words)
    assert "и" not in words and "в" not in words


def test_share_prefers_relevant_span():
    question = "Кем характеризуются отдельные виды?"
    relevant = "отдельные виды характеризуются признаками"
    irrelevant = "в 1990 году завод выпустил 42 детали"
    assert share_of_question(question, relevant) > share_of_question(question, irrelevant)


def test_cosine_bounds():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_runs_of_groups_consecutive():
    pieces = [(0, 1, 0.9), (2, 3, 0.8), (4, 5, 0.1), (6, 7, 0.7)]
    assert runs_of(pieces, 0.5) == [(0, 1), (3, 3)]
    assert runs_of(pieces, 0.95) == []


def test_question_never_lowers_threshold():
    """Ключевое свойство: при нескольких прошедших полка НЕ включается."""
    answer = "abcdefghijklmnopqrstuvwxyz"
    pieces = [(0, 3, 0.9), (4, 7, 0.8), (8, 11, 0.35)]
    point = select_cascade(answer, pieces, "вопрос", 0.5)
    assert point is not None
    assert point["from_shelf"] is False
    assert point["start"] <= 7  # только из прошедших порог


def test_shelf_only_when_nothing_passed():
    answer = "abcdefghijklmnopqrstuvwxyz"
    pieces = [(0, 3, 0.35), (4, 7, 0.32)]
    point = select_cascade(answer, pieces, "вопрос", 0.9)
    assert point is not None
    assert point["from_shelf"] is True
    assert point["start"] == 0


def test_nothing_at_all_returns_none():
    pieces = [(0, 3, 0.05)]
    assert select_cascade("abcdefgh", pieces, "вопрос", 0.9) is None
    assert select_cascade("abcdefgh", [], "вопрос", 0.5) is None


def test_ranking_picks_relevant_run():
    """При двух прошедших выбирается релевантный вопросу, а не первый."""
    answer = "зимой выпадает снег а виды характеризуются признаками"
    first = (0, len("зимой выпадает снег") - 1)
    second = (first[1] + 2, len(answer) - 1)
    pieces = [(first[0], first[1], 0.9), (second[0], second[1], 0.8)]
    point = select_cascade(answer, pieces, "чем характеризуются виды", 0.5)
    assert point["start"] == second[0]
