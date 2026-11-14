import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spanver.candidates import build_candidates

TYPES = ("contradiction", "number", "entity", "omission", "unsupported_addition", "temporal", "scope")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Вычитка 200 пар</title>
<style>
body{font-family:Georgia,serif;margin:20px;background:#fafafa;color:#111}
h1{font-size:20px}
.pair{border:1px solid #ccc;border-radius:6px;padding:10px;margin:12px 0;background:#fff}
.meta{font-size:12px;color:#666;margin-bottom:6px}
.q{font-weight:bold;margin-bottom:6px}
.out{line-height:1.6;white-space:pre-wrap}
.mark{cursor:pointer;padding:1px 0;border-bottom:2px dotted #c60}
.mark.gold{background:#cfe8cf;border-bottom:2px solid #2a7a2a}
.mark.rej{background:#eee;color:#999;text-decoration:line-through;border-bottom:2px solid #999}
.panel{margin-top:8px;padding:8px;border-top:1px dashed #ccc;font-size:13px}
.panel button{margin:2px;padding:3px 6px;cursor:pointer}
.bar{position:sticky;bottom:0;background:#fafafa;padding:10px 0;border-top:1px solid #ddd}
</style>
</head>
<body>
<h1>Вычитка пар: клик по подсвеченному фрагменту, затем тип или отклонение</h1>
<div id="status"></div>
<div id="list"></div>
<div class="bar"><button id="save">Скачать JSON</button></div>
<script>
const DATA = __DATA__;
const state = {};
function esc(text){return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function key(span){return span.start + "-" + span.end;}
function markClass(pairId, span){
  const item = (state[pairId] || {})[key(span)];
  if(!item) return "mark";
  return item.status === "gold" ? "mark gold" : "mark rej";
}
function renderOutput(pair){
  const spans = pair.detector_spans.slice().sort((a,b)=>a.start-b.start);
  let html = "";
  let cursor = 0;
  spans.forEach((span, index)=>{
    if(span.start < cursor || span.end >= pair.model_output.length) return;
    html += esc(pair.model_output.slice(cursor, span.start));
    html += '<span class="' + markClass(pair.pair_id, span) + '" data-pair="' + pair.pair_id + '" data-index="' + index + '">' + esc(pair.model_output.slice(span.start, span.end + 1)) + '</span>';
    cursor = span.end + 1;
  });
  html += esc(pair.model_output.slice(cursor));
  return html;
}
function render(){
  const list = document.getElementById("list");
  list.innerHTML = "";
  DATA.forEach((pair)=>{
    const box = document.createElement("div");
    box.className = "pair";
    box.innerHTML = '<div class="meta">' + esc(pair.pair_id) + ' · ' + esc(pair.domain) + ' · score ' + pair.detector_spans[0].score + '</div>' +
      '<div class="q">' + esc(pair.query_text) + '</div>' +
      '<div class="out">' + renderOutput(pair) + '</div>' +
      '<div class="panel" id="panel-' + pair.pair_id + '"></div>';
    list.appendChild(box);
  });
  document.querySelectorAll(".mark").forEach((node)=>{
    node.addEventListener("click", ()=>openPanel(node.dataset.pair, Number(node.dataset.index)));
  });
  const done = Object.values(state).reduce((acc, item)=>acc + Object.keys(item).length, 0);
  document.getElementById("status").textContent = "отмечено фрагментов: " + done;
}
function openPanel(pairId, index){
  const pair = DATA.find((item)=>item.pair_id === pairId);
  const span = pair.detector_spans.slice().sort((a,b)=>a.start-b.start)[index];
  if(!span) return;
  const panel = document.getElementById("panel-" + pairId);
  panel.innerHTML = "";
  const label = document.createElement("div");
  label.textContent = "Фрагмент: " + span.text.slice(0, 140);
  panel.appendChild(label);
  const typeNames = __TYPES__;
  typeNames.forEach((type)=>{
    const button = document.createElement("button");
    button.textContent = type;
    button.addEventListener("click", ()=>{
      state[pairId] = state[pairId] || {};
      state[pairId][key(span)] = {status: "gold", type: type};
      render();
    });
    panel.appendChild(button);
  });
  const reject = document.createElement("button");
  reject.textContent = "отклонить";
  reject.addEventListener("click", ()=>{
    state[pairId] = state[pairId] || {};
    state[pairId][key(span)] = {status: "rejected"};
    render();
  });
  panel.appendChild(reject);
}
document.getElementById("save").addEventListener("click", ()=>{
  const result = DATA.map((pair)=>({
    pair_id: pair.pair_id,
    query_text: pair.query_text,
    model_output: pair.model_output,
    spans: pair.detector_spans.map((span)=>({text: span.text, start: span.start, end: span.end})),
    gold: (state[pair.pair_id] ? Object.keys(state[pair.pair_id]) : []).filter((k)=>state[pair.pair_id][k].status === "gold").map((k)=>{
      const span = pair.detector_spans.find((item)=>key(item) === k);
      const chosen = state[pair.pair_id][k];
      return {text: span.text, start: span.start, end: span.end, type: chosen.type};
    })
  }));
  const blob = new Blob([result.map((item)=>JSON.stringify(item)).join("\\n") + "\\n"], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "candidates_200_check.jsonl";
  link.click();
});
render();
</script>
</body>
</html>
"""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_html(pairs):
    payload = []
    for pair, spans, best in pairs:
        payload.append(
            {
                "pair_id": pair["pair_id"],
                "domain": pair["domain"],
                "query_text": pair["query_text"],
                "model_output": pair["model_output"],
                "detector_spans": spans,
            }
        )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__DATA__", data)
        .replace("__TYPES__", json.dumps(list(TYPES), ensure_ascii=False))
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)
    out = Path(args.out)
    gold = read_jsonl(out / "gold.jsonl")
    unlabeled = read_jsonl(out / "pairs_unlabeled.jsonl")
    pairs, stats = build_candidates(gold, unlabeled, limit=args.limit)
    rows = []
    checks = []
    for pair, spans, best in pairs:
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "domain": pair["domain"],
                "query_id": pair["query_id"],
                "query_text": pair["query_text"],
                "model_output": pair["model_output"],
                "doc_ids": pair["doc_ids"],
                "detector_spans": spans,
            }
        )
        checks.append(
            {
                "pair_id": pair["pair_id"],
                "query_text": pair["query_text"],
                "model_output": pair["model_output"],
                "spans": [{"text": span["text"], "start": span["start"], "end": span["end"]} for span in spans],
                "gold": [],
            }
        )
    candidates_path = out / "candidates_200.jsonl"
    check_path = out / "candidates_200_check.jsonl"
    html_path = out / "candidates_200.html"
    write_jsonl(candidates_path, rows)
    write_jsonl(check_path, checks)
    html_path.write_text(render_html(pairs), encoding="utf-8")
    stats["candidates_sha256"] = sha256_file(candidates_path)
    stats["check_sha256"] = sha256_file(check_path)
    stats["html_sha256"] = sha256_file(html_path)
    (out / "candidates_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pool = {stats['pool']}")
    print(f"candidates = {stats['count']}")
    print(f"doc overlap with gold = {stats['doc_overlap_with_gold']}")
    print(f"max per doc = {stats['max_per_doc']}")
    print(f"sha256 candidates_200.jsonl = {stats['candidates_sha256']}")
    print(f"sha256 candidates_200_check.jsonl = {stats['check_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
