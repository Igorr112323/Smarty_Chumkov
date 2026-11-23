"""Синтетические проверки фиксированного отбора inner без доступа к данным."""
from tools.recalculate67_b import grid_points
from tools.summarize59 import pick_threshold


def point(start,tp,fp,fn,negative):
    support={'token_tp':tp,'token_fp':fp,'token_fn':fn,'negative_tokens':negative,
             'flagged_negative_tokens':fp}
    return {'start':start,'extend':start/2,'support':support}


def test_max_f1_among_eligible_only():
    grid=[point(.1, 90,60,10,100),point(.2,70,40,30,100),point(.3,50,10,50,100)]
    points=grid_points([grid,grid,grid,grid])
    assert points[0]['fpr_token'] == .6
    assert points[1]['fpr_token'] == .4
    assert pick_threshold([grid,grid,grid,grid])['start']==.2


def test_no_eligible_threshold():
    grid=[point(.1,90,60,10,100),point(.2,80,41,20,100)]
    assert pick_threshold([grid]*4) is None
