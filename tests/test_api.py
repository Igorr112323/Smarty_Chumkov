import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from spanver.api import Handler


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_verify_endpoint():
    server = start_server()
    try:
        port = server.server_address[1]
        body = json.dumps({"answer": "В 1990 году завод выпустил 1200 деталей.", "document": "Завод."}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/verify", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert set(payload) == {"spans", "ai_share", "meta"}
        assert isinstance(payload["spans"], list)
        assert 0.0 <= payload["ai_share"] <= 1.0
    finally:
        server.shutdown()
        server.server_close()


def test_health_endpoint():
    server = start_server()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_path():
    server = start_server()
    try:
        port = server.server_address[1]
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=20)
            assert False
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_root_serves_local_ui():
    server = start_server()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=20) as response:
            assert response.headers["Content-Type"].startswith("text/html")
            page = response.read().decode("utf-8")
        assert "Проверить" in page
        assert 'id="answer"' in page
        assert 'id="doc"' in page
        assert 'fetch(\'/verify\'' in page
        assert "http://" not in page and "https://" not in page
        assert "<link" not in page
    finally:
        server.shutdown()
        server.server_close()
