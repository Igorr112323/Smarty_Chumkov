import pytest
from tools.summarize63c import average_candidates
from tools.train63 import decide


def test_averaging_by_boundaries_and_selection_by_risk():
    reports = [{'inner_candidates': {'a': [
        {'start': 0, 'end': 2, 'risk': 0.6}, {'start': 3, 'end': 5, 'risk': 0.9}]}},
        {'inner_candidates': {'a': [
            {'start': 3, 'end': 5, 'risk': 0.7}, {'start': 0, 'end': 2, 'risk': 0.4}]}}]
    avg = average_candidates(reports, 'inner_candidates')['a']
    assert avg == [{'start': 0, 'end': 2, 'risk': 0.5}, {'start': 3, 'end': 5, 'risk': 0.8}]
    assert decide(avg, 0.5, limit=1) == [{'start': 3, 'end': 5}]


def test_reject_missing_candidate():
    reports = [{'outer_candidates': {'a': [{'start': 0, 'end': 1, 'risk': 0.5}]}},
               {'outer_candidates': {'a': []}}]
    with pytest.raises(ValueError, match='разные кандидаты'):
        average_candidates(reports, 'outer_candidates')
