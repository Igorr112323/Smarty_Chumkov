"""ПРОМПТ 59: вопрос на входе — обратная совместимость и сама семантика.

Проверяется три вещи:
1. без флага (`QUESTION_INPUT = False`) вход и признаки совпадают с прежними —
   это условие контрольного прогона A;
2. с флагом вопрос попадает в первый сегмент, а `doc_overlap` считается по
   `question + document`;
3. пустой вопрос ничего не ломает: вход вырождается в прежний.
"""
import pytest

from spanver import lexfeat


@pytest.fixture(autouse=True)
def reset_flag():
    yield
    lexfeat.with_question(False)


def pair(**extra):
    payload = {
        "pair_id": "p1",
        "query_text": "Кем характеризуются виды?",
        "document_text": "Виды характеризуются отдельными признаками.",
        "model_output": "Виды характеризуются 42 признаками.",
    }
    payload.update(extra)
    return payload


def test_flag_defaults_to_off():
    assert lexfeat.QUESTION_INPUT is False


def test_without_flag_segment_is_chunk_only():
    lexfeat.with_question(False)
    assert lexfeat.segment_text(pair(), "чанк") == "чанк"
    assert lexfeat.feature_document(pair()) == "Виды характеризуются отдельными признаками."


def test_with_flag_question_is_in_first_segment():
    lexfeat.with_question(True)
    text = lexfeat.segment_text(pair(), "чанк")
    assert text.startswith("Кем характеризуются виды?")
    assert "чанк" in text


def test_with_flag_overlap_counts_question():
    """Токен, встречающийся только в вопросе, становится подтверждённым."""
    document = "совершенно другой текст"
    lexfeat.with_question(False)
    off = lexfeat.answer_features("характеризуются", document)
    lexfeat.with_question(True)
    on = lexfeat.answer_features("характеризуются", lexfeat.feature_document(pair(document_text=document)))
    assert off and on
    assert off[0][2][4] == 0.0
    assert on[0][2][4] == 1.0


def test_empty_question_degrades_to_previous_input():
    lexfeat.with_question(False)
    before = lexfeat.segment_text(pair(query_text=""), "чанк")
    lexfeat.with_question(True)
    after = lexfeat.segment_text(pair(query_text=""), "чанк")
    assert before == after == "чанк"


def test_empty_question_keeps_features_identical():
    lexfeat.with_question(False)
    before = lexfeat.feature_document(pair(query_text=""))
    lexfeat.with_question(True)
    after = lexfeat.feature_document(pair(query_text=""))
    assert before == after


def test_flag_returns_new_state():
    assert lexfeat.with_question(True) is True
    assert lexfeat.with_question(False) is False


def test_question_transferred_for_all_pairs():
    """699 из 699 пар train/val имеют непустой вопрос (стадия 1.2)."""
    from tools.preflight_13 import our_rows

    rows = our_rows()
    assert len(rows) == 699
    missing = [row["pair_id"] for row in rows if not (row.get("query_text") or "").strip()]
    assert missing == [], f"без вопроса: {len(missing)}"
