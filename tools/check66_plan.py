"""Fail closed перед запуском 50 обучающих jobs; читает только 699 и сохранённые гейты."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from spanver.evaluate import read_jsonl
from tools.preflight_13 import id_digest


def main():
    a=ROOT/'artifacts';gold=a/'gold66_trainval.jsonl';sha=hashlib.sha256(gold.read_bytes()).hexdigest()
    expected='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff'
    pairs=read_jsonl(gold)
    if sha!=expected or len(pairs)!=699 or id_digest(pairs)!='22fd4f1dc99bb96c59ed644725690be9d7d7f5fbeac7cb1189b7bc237c088da1':raise ValueError('неверный 699 gold')
    mem=json.loads((a/'memory66_step.json').read_text());length=json.loads((a/'length65_check.json').read_text());anchor=json.loads((a/'anchor65_check.json').read_text());threshold=json.loads((a/'threshold65_check.json').read_text())
    if not mem.get('memory_pass') or not mem.get('forward_backward_optimizer_step') or mem.get('weights_saved') or mem.get('gold_sha256')!=sha or mem.get('sequence_width')!=512 or mem.get('micro_batch')!=4 or mem.get('batch')!=16:raise ValueError('память не проверена')
    if not length.get('candidate_loss_check_pass') or length.get('total')!=5973 or length.get('gold_sha256')!=sha:raise ValueError('потеря кандидатов')
    if not anchor.get('all_pass'):raise ValueError('якорь A не подтверждён')
    if len(threshold.get('folds',[]))!=10 or not all(f.get('rule_passed') for f in threshold['folds']):raise ValueError('правило порога не подтверждено')
    print('Данные и проверенные гейты готовы; ρ текущей модели диагностируется после обучения, не блокирует.')
    return 0
if __name__=='__main__':raise SystemExit(main())
