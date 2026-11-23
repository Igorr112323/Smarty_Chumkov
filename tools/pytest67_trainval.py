"""Plugin ограничивает тест наличия вопросов 699 сохранёнными train/val.
Это НЕ тест чтения CSV: исходный Dataset исключён во избежание чтения test.
"""
import hashlib
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent.parent

@pytest.fixture(autouse=True)
def question_trainval_fixture(request,monkeypatch):
    if request.node.name != 'test_question_transferred_for_all_pairs':
        return
    path=ROOT/'artifacts/gold66_trainval.jsonl'
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff':
        raise ValueError('train/val SHA mismatch')
    rows=[json.loads(line) for line in data.splitlines() if line]
    if len(rows)!=699 or len({x['pair_id'] for x in rows})!=699:
        raise ValueError('train/val ID count')
    import tools.preflight_13 as preflight
    monkeypatch.setattr(preflight,'our_rows',lambda:rows)
