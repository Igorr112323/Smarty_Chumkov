import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .model_features import ModelUnavailable
from .model_score import HeadMissing, verify_model
from .module import verify

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spanver — проверка ответа по документу</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 960px; padding: 24px 16px 64px; line-height: 1.45; }
h1 { font-size: 20px; margin: 0 0 4px; }
p.hint { margin: 0 0 20px; opacity: .7; font-size: 14px; }
label { display: block; font-size: 14px; font-weight: 600; margin: 16px 0 6px; }
textarea { width: 100%; min-height: 130px; box-sizing: border-box; padding: 10px; font: 14px/1.4 ui-monospace, monospace; border: 1px solid #8884; border-radius: 8px; resize: vertical; }
#doc { min-height: 170px; }
button { margin-top: 16px; padding: 10px 22px; font-size: 15px; border: 0; border-radius: 8px; background: #2f6fed; color: #fff; cursor: pointer; }
button:disabled { opacity: .55; cursor: progress; }
#status { margin-left: 12px; font-size: 14px; opacity: .75; }
#summary { margin: 20px 0 8px; font-size: 15px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #8883; vertical-align: top; }
th { font-weight: 600; opacity: .75; }
.frag { font-family: ui-monospace, monospace; }
mark { background: #ffd16688; border-radius: 3px; padding: 0 2px; }
.hidden { display: none; }
</style>
</head>
<body>
<h1>Проверка ответа по документу</h1>
<p class="hint">Модуль ищет в ответе фрагменты без опоры в документе: координаты, риск, тип.</p>
<label for="answer">Ответ</label>
<textarea id="answer" placeholder="Вставьте текст ответа модели"></textarea>
<label for="doc">Документ</label>
<textarea id="doc" placeholder="Вставьте исходный документ"></textarea>
<div>
<button id="run" type="button">Проверить</button>
<span id="status"></span>
</div>
<div id="summary" class="hidden"></div>
<div id="result" class="hidden">
<table>
<thead><tr><th>№</th><th>start–end</th><th>риск</th><th>тип</th><th>confidence</th><th>фрагмент</th></tr></thead>
<tbody id="rows"></tbody>
</table>
</div>
<script>
const answerBox = document.getElementById('answer');
const docBox = document.getElementById('doc');
const runButton = document.getElementById('run');
const statusLine = document.getElementById('status');
const summary = document.getElementById('summary');
const result = document.getElementById('result');
const rows = document.getElementById('rows');

function cell(text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) { td.className = className; }
  return td;
}

async function run() {
  const answer = answerBox.value;
  const document = docBox.value;
  if (!answer.trim()) {
    statusLine.textContent = 'нужен текст ответа';
    return;
  }
  runButton.disabled = true;
  statusLine.textContent = 'идёт проверка…';
  summary.classList.add('hidden');
  result.classList.add('hidden');
  rows.replaceChildren();
  try {
    const response = await fetch('/verify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({answer, document})
    });
    if (!response.ok) { throw new Error('HTTP ' + response.status); }
    const data = await response.json();
    const spans = data.spans || [];
    summary.textContent = 'Доля участия ИИ: ' + data.ai_share.toFixed(4) +
      ' · фрагментов: ' + spans.length +
      ' · ответ ' + data.meta.answer_len + ' зн., документ ' + data.meta.doc_len +
      ' зн., ' + data.meta.elapsed_ms + ' мс';
    summary.classList.remove('hidden');
    spans.forEach((span, index) => {
      const tr = document.createElement('tr');
      tr.appendChild(cell(String(index + 1)));
      tr.appendChild(cell(span.start + '–' + span.end));
      tr.appendChild(cell(span.risk.toFixed(4)));
      tr.appendChild(cell(span.type));
      tr.appendChild(cell(span.confidence.toFixed(4)));
      const text = answer.slice(span.start, span.end + 1);
      const td = document.createElement('td');
      td.className = 'frag';
      const mark = document.createElement('mark');
      mark.textContent = text;
      td.appendChild(mark);
      tr.appendChild(td);
      rows.appendChild(tr);
    });
    result.classList.remove('hidden');
    statusLine.textContent = spans.length ? 'готово' : 'готово, подозрительных фрагментов нет';
  } catch (error) {
    statusLine.textContent = 'ошибка: ' + error.message;
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener('click', run);

[answerBox, docBox].forEach((box) => {
  box.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { run(); }
  });
});
</script>
</body>
</html>
"""



class Handler(BaseHTTPRequestHandler):
    backend = "lexical"
    weights = None
    head = "artifacts/model_head.json"

    def payload(self, answer, document):
        if self.backend == "model":
            if not self.weights:
                raise ModelUnavailable("укажите каталог весов: --weights models\\ruSciBERT")
            return verify_model(answer, document, self.weights, self.head)
        return verify(answer, document)

    def send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, code, text):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_html(200, PAGE)
        elif path == "/health":
            self.send_json(200, {"status": "ok", "service": "spanver"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/verify":
            self.send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            answer = payload.get("answer") or ""
            document = payload.get("document") or ""
        except (ValueError, UnicodeDecodeError):
            self.send_json(400, {"error": "invalid json"})
            return
        try:
            result = self.payload(answer, document)
        except (ModelUnavailable, HeadMissing) as error:
            self.send_json(503, {"error": str(error)})
            return
        self.send_json(200, result)

    def log_message(self, format, *args):
        return


def serve(port=8765, host="0.0.0.0", backend="lexical", weights=None, head="artifacts/model_head.json"):
    Handler.backend = backend
    Handler.weights = weights
    Handler.head = head
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"listening on http://localhost:{port} (backend {backend})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
