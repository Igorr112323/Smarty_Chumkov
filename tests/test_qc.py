from spanver.qc import has_control, inspect, match_steps


def pair(spans, document="документ", output="ответ модели"):
    return {"document_text": document, "model_output": output, "spans": spans}


def test_clean_pair_is_ok():
    result = inspect([pair([{"text": "модели", "start": 6, "end": 11}])])
    assert result["ok"] is True
    assert result["coverage"] == 1.0
    assert result["out_of_bounds"] == 0
    assert result["span_text_mismatch"] == 0


def test_out_of_bounds_detected():
    result = inspect([pair([{"text": "модели", "start": 6, "end": 999}])])
    assert result["out_of_bounds"] == 1
    assert result["ok"] is False


def test_mismatch_detected():
    result = inspect([pair([{"text": "другой текст", "start": 0, "end": 11}])])
    assert result["span_text_mismatch"] == 1
    assert result["ok"] is False


def test_nested_is_counter_not_error():
    result = inspect([pair([{"text": "ответ модели", "start": 0, "end": 11}, {"text": "ответ", "start": 0, "end": 4}])])
    assert result["nested"] == 1
    assert result["overlaps"] == 0
    assert result["ok"] is True
    assert result["coverage"] == 1.0


def test_overlap_is_counter_not_error():
    result = inspect([pair([{"text": "ответ мод", "start": 0, "end": 8}, {"text": "модели", "start": 6, "end": 11}])])
    assert result["overlaps"] == 1
    assert result["ok"] is True


def test_control_characters_counted():
    result = inspect([pair([], document="документ\x00с")])
    assert result["control_pairs"] == 1
    assert has_control("текст\x07") is True
    assert has_control("текст\nтекст") is False


def test_match_steps_normalization():
    assert match_steps("он сказал \u00abда\u00bb", "он сказал \"да\"", 0, 13) == 3
    assert match_steps("совсем другое", "ответ модели", 0, 4) is None
