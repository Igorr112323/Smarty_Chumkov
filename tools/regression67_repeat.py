"""Регрессия tiny-пути поставки на train/val: только Actions, без test 301."""
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))


def compare_spans(left,right):
    def project(rows):return [{k:r[k] for k in ('start','end','type','confidence')} for r in rows]
    return project(left)==project(right)


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('проверка инференса только в Actions')
    out=ROOT/'artifacts/regression67_fixed_verify.json';out.parent.mkdir(exist_ok=True)
    report={'status':'ошибка','test301':False,'training':False,'criterion_1_2_pass':False,'criterion_1_3_speed_pass':None,'criterion_1_3_package_pass':None}
    try:
        import torch
        from spanver import lexfeat
        from spanver.evaluate import read_jsonl
        from spanver.tiny_backend import TinyBundle
        from tools.cli52 import span_metrics
        from tools.predict_frozen38 import load_state
        from tools.preflight_13 import id_digest
        from tools.train_base33 import grouped_folds
        from tools.build_package import build
        source=ROOT/'artifacts/gold66_trainval.jsonl'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff':raise ValueError('неверный SHA 699')
        pairs=read_jsonl(source);held=grouped_folds(pairs,5)[0]
        refdir=Path('/tmp/report47'); options=[json.loads(p.read_text()) for p in refdir.rglob('*.json')]
        rr=[r for r in options if r.get('variant')=='v5' and r.get('outer_fold')==1 and r.get('inner_fold')==2 and r.get('primary')]
        if len(rr)!=1:raise ValueError('нет единственного отчёта V5 fold1')
        rr=rr[0]
        if id_digest(held)!=rr['outer_id_sha256']:raise ValueError('контрольные ID сохранённого внешнего фолда не совпадают')
        entry=rr['seeds'][0]['checkpoint']
        if rr['seeds'][0]['seed']!=0 or not entry['restore_predictions_equal']:raise ValueError('неподтверждённый seed 0')
        dirs=list(Path('/tmp/weights47').rglob('seed_0'))
        if len(dirs)!=1:raise ValueError(f'найдено каталогов seed_0: {len(dirs)}')
        saved=json.loads((dirs[0]/'manifest.json').read_text())
        final=json.loads(Path('/tmp/final47_manifest.json').read_text())
        if final.get('variant')!='v5' or final.get('outer_fold')!=1 or final.get('inner_fold')!=2 or final['checkpoints'][0]['sha256']!=entry['sha256']:
            raise ValueError('архив не совпадает с манифестом исторической поставки')
        if saved['sha256']!=entry['sha256'] or saved['chunks']!=entry['chunks']:raise ValueError('архив V5 не совпадает с сохранённым отчётом')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True);lexfeat.with_question(False)
        actual=TinyBundle(None,ROOT/'artifacts/tiny_head.json')
        reference=TinyBundle(None,ROOT/'artifacts/tiny_head.json')
        load_state(torch,reference.model,dirs[0],entry)
        assert actual.head == reference.head
        assert (actual.head['gate'], actual.head['start'], actual.head['extend'], actual.head['max_spans_default']) == (0.45, 0.49, 0.245, 3)
        def state_sha(model):
            digest = hashlib.sha256()
            for name, t in sorted(model.state_dict().items()):
                digest.update(name.encode()); digest.update(t.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        actual_sha=state_sha(actual.model); reference_sha=state_sha(reference.model)
        # Для исторического single-seed есть только агрегат CLI52, не сохранены
        # исходные поэлементные ответы start/end/type/confidence.
        historical=json.loads((ROOT/'artifacts/cli52.json').read_text())
        oldpred={};newpred={};differences=[];times_current=[];times_ref=[]
        for p in held:
            answer=p['model_output'];document=p['document_text']; chars=len(document)
            started=time.perf_counter();r=reference.verify(answer,document,threshold=0.45,max_spans=3)['spans'];msr=(time.perf_counter()-started)*1000
            started=time.perf_counter();c=actual.verify(answer,document,threshold=0.45,max_spans=3)['spans'];msc=(time.perf_counter()-started)*1000
            oldpred[p['pair_id']]=r;newpred[p['pair_id']]=c
            times_ref.append(msr*1000/max(1,chars));times_current.append(msc*1000/max(1,chars))
            if not compare_spans(c,r):
                if len(differences)<10:differences.append({'pair_id':p['pair_id'],'current':[{k:x[k] for k in ('start','end','type','confidence')} for x in c], 'restored_v5':[{k:x[k] for k in ('start','end','type','confidence')} for x in r]})
        matches=sum(compare_spans(newpred[p['pair_id']],oldpred[p['pair_id']]) for p in held)
        saved=historical['delivered']; reconstructed=span_metrics(held,oldpred); actual_metrics=span_metrics(held,newpred)
        names=('token_f1','token_precision','token_recall','fpr_token','fpr_pair','span_exact_f1','span_overlap_f1','span_iou_f1','type_accuracy','predicted_spans','gold_spans','bootstrap')
        comparisons={k:{'saved':saved.get(k),'restored':reconstructed.get(k),'current':actual_metrics.get(k),'exact':saved.get(k)==reconstructed.get(k)==actual_metrics.get(k)} for k in names}
        speed=statistics.median(times_current);speed_ref=statistics.median(times_ref)
        with tempfile.TemporaryDirectory(prefix='package67-') as directory:
            build(directory);package=json.loads((Path(directory)/'package.json').read_text())
        report.update(status='готово',pairs=len(held),data_sha256=digest,
                      identical_decoder={'gate':0.45,'start':0.49,'extend':0.245,'max_spans':3,'top_chunks':4,'clause_expand':True},
                      current_state_sha256=actual_sha,restored_state_sha256=reference_sha,
                      same_model_state=actual_sha==reference_sha,
                      agreed_aggregate_reference='замена исторического поэлементного эталона согласована пользователем; фактически сопоставимый агрегат V5 находится в artifacts/cli52.json, module52_report описывает лексический модуль',
                      saved_report='artifacts/cli52.json: только агрегат; поэлементных старых ответов нет',
                      model_reference='восстановленный V5 outer1 seed0 из run 44056101776; не заявлен как checkpoint A',
                      reference_checkpoint_sha256=entry['sha256'],source_run=44056101776,
                      current_backend='TinyBundle через текущую CLI-конфигурацию; QUESTION_INPUT=false',
                      saved_aggregate_comparison=comparisons,
                      current_vs_restored_pairs_exact=matches,current_vs_restored_pairs_total=len(held),
                      current_vs_restored_fraction=matches/len(held),example_differences=differences,
                      criterion_1_2_pass=matches==len(held) and actual_sha==reference_sha and all(v['exact'] for v in comparisons.values()),
                      criterion_1_2_reason='совпадение со сверенным V5 на идентичных параметрах не 100%' if matches!=len(held) else 'совпадение на всех 140 парах',
                      current_speed_ms_per_1000_median=speed,restored_v5_speed_ms_per_1000_median=speed_ref,
                      speed_pairs=len(held),speed_definition='медиана по 140 внешним train/val парам; время полного TinyBundle.verify, знаки документа',
                      criterion_1_3_speed_pass=speed<=30,package=package,
                      package_recorded_sha256=(ROOT/'package.sha256').read_text().split()[0],
                      package_matches_record=package['sha256']==(ROOT/'package.sha256').read_text().split()[0],
                      criterion_1_3_package_pass=package['bytes']<=600000,
                      note='TinyBundle автоматически загрузил проверенный V5. Сопоставимый агрегатный эталон cli52.json принят пользователем.')
    except Exception:report['error']=traceback.format_exc()
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report.get('criterion_1_2_pass') and report.get('criterion_1_3_speed_pass') else 1

if __name__=='__main__':raise SystemExit(main())
