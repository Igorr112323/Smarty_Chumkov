from spanver.types import base_type, classify, features


def test_contradiction():
    assert classify("столица находится на берегу", "Contradiction") == "contradiction"


def test_number():
    assert classify("в 1861 году отменили крепостное право", "Contradiction") == "number"
    assert classify("сорок два человека пришли", "Partial") == "number"


def test_temporal():
    assert classify("ранее это правило не применялось", "Partial") == "temporal"


def test_entity():
    assert classify("он встретил Ивана Петрова в городе", "Unconfirmed") == "entity"


def test_scope():
    assert classify("каждый сотрудник обязан отчитаться", "Unconfirmed") == "scope"


def test_omission():
    assert classify("этот факт отсутствует в тексте", "Missing") == "omission"


def test_unsupported_addition():
    assert classify("подробность добавлена без опоры", "Unconfirmed") == "unsupported_addition"
    assert classify("подробность добавлена без опоры", "Excess") == "unsupported_addition"
    assert classify("подробность добавлена без опоры", "Oversight") == "unsupported_addition"
    assert classify("подробность добавлена без опоры", "Partial") == "unsupported_addition"


def test_priority_number_over_scope():
    assert classify("все 5 студентов сдали экзамен", "Unconfirmed") == "number"


def test_contradiction_with_number_becomes_number():
    assert classify("в 1990 году завод закрыли", "Contradiction") == "number"
    assert base_type("Contradiction") == "contradiction"
    assert "temporal" in features("в 1990 году завод закрыли")
