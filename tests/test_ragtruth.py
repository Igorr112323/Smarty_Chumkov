from spanver.ragtruth import chunks_from, context_text, doc_id, rebuild, select_pairs, spans_from_labels, src_sha1

TEXT = "Paris is the capital. It was founded in 250 BC and has 2.1 million people."


def test_spans_from_labels_converts_to_inclusive():
    labels = [{"start": 0, "end": 5, "label_type": "Evident Conflict"}]
    spans = spans_from_labels(labels)
    assert spans == [{"start": 0, "end": 4, "label_type": "Evident Conflict"}]


def test_spans_merge_overlaps_and_join_types():
    labels = [
        {"start": 22, "end": 46, "label_type": "Evident Conflict"},
        {"start": 30, "end": 52, "label_type": "Evident Baseless Info"},
    ]
    spans = spans_from_labels(labels)
    assert len(spans) == 1
    assert spans[0]["start"] == 22
    assert spans[0]["end"] == 51
    assert "Evident Conflict" in spans[0]["label_type"]
    assert "Evident Baseless Info" in spans[0]["label_type"]


def test_chunks_cover_text_without_gaps():
    spans = spans_from_labels([{"start": 28, "end": 34, "label_type": "Evident Conflict"}])
    chunks = chunks_from(TEXT, spans)
    assert "".join(chunk["text"] for chunk in chunks) == TEXT
    assert [chunk["kind"] for chunk in chunks] == ["text", "span", "text"]


def test_rebuild_keeps_coordinates_exact_with_longer_translation():
    spans = spans_from_labels([{"start": 0, "end": 17, "label_type": "Evident Conflict"}])
    chunks = chunks_from(TEXT, spans)
    translated = [chunk["text"].upper().replace("PARIS", "ПАРИЖ-ГОРОД") for chunk in chunks]
    text, new_spans = rebuild(chunks, translated)
    assert text == "".join(translated)
    for span in new_spans:
        assert 0 <= span["start"] <= span["end"] < len(text)
        assert text[span["start"] : span["end"] + 1] == span["span_text"]


def test_rebuild_handles_empty_translation_of_prefix():
    spans = spans_from_labels([{"start": 10, "end": 14, "label_type": "Subtle Conflict"}])
    chunks = chunks_from(TEXT, spans)
    translated = ["" if chunk["kind"] == "text" else chunk["text"] for chunk in chunks]
    text, new_spans = rebuild(chunks, translated)
    assert text == "".join(translated)
    assert len(new_spans) == len(spans)


def test_doc_id_and_sha1_are_stable():
    assert doc_id("text") == doc_id("text")
    assert len(doc_id("text")) == 16
    assert src_sha1("text") == src_sha1("text")
    assert len(src_sha1("text")) == 40


def test_context_text_handles_dict_and_string():
    assert context_text({"source_info": "abc"}) == "abc"
    assert context_text({"source_info": {"name": "Kitchen"}}) == '{"name": "Kitchen"}'
    assert context_text({"prompt": "q"}) == "q"
    assert context_text({}) == ""


def test_select_pairs_balances_and_limits():
    rows = []
    types = ("Evident Conflict", "Evident Baseless Info", "Subtle Baseless Info")
    for index in range(30):
        rows.append({"id": str(index), "response": "text", "labels": [{"start": 0, "end": 3, "label_type": types[index % 3]}]})
    for index in range(30, 40):
        rows.append({"id": str(index), "response": "clean", "labels": []})
    positives, negatives = select_pairs(rows, limit=9, negatives=3)
    assert len(positives) == 9
    assert len(negatives) == 3
    assert all(spans for _, spans in positives)


def test_split_parts_respects_limit_and_keeps_text():
    from tools.translate_ragtruth import MAX_PART, split_parts

    text = ("word " * 2000).strip()
    parts = split_parts(text)
    assert parts
    assert all(len(part) <= MAX_PART for part in parts)
    assert "".join(parts) == text


def test_split_parts_handles_short_text():
    from tools.translate_ragtruth import split_parts

    assert split_parts("hello") == ["hello"]
    assert split_parts("") == []


def test_channel_order_matches_contract():
    from tools.translate_ragtruth import CHANNEL_ORDER

    assert CHANNEL_ORDER == ("dict-chrome-ex", "simplytranslate", "mymemory", "lingva")
