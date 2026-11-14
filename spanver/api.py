import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .module import verify


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path in ("/health", "/"):
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
        self.send_json(200, verify(answer, document))

    def log_message(self, format, *args):
        return


def serve(port=8765, host="0.0.0.0"):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"listening on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
