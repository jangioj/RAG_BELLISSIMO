"""Local same-origin HTTP bridge for the existing src.rag_backend.ask_rag.

Run from the RAG repository root after copying this file and dist/ there:
    python http_adapter.py
"""

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
SESSION_TTL = 12 * 60 * 60
MAX_BODY = 32 * 1024
_sessions = {}
_sessions_lock = threading.Lock()


@dataclass
class Session:
    api_key: str | None = None
    history: list = field(default_factory=list)
    conversation_id: str = field(default_factory=lambda: secrets.token_urlsafe(18))
    touched: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)


def _source_list(value):
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    output = []
    for source in value:
        if isinstance(source, str):
            output.append({"title": source[:160]})
            continue
        if not isinstance(source, dict):
            continue
        title = source.get("title") or source.get("source") or source.get("document") or source.get("type") or "Source"
        locator = source.get("locator") or source.get("page") or source.get("id") or ""
        excerpt = source.get("excerpt") or source.get("snippet") or source.get("text") or ""
        output.append({
            "title": str(title)[:160],
            "locator": str(locator)[:100],
            "excerpt": str(excerpt)[:600],
        })
    return output[:30]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def log_message(self, format, *args):
        # No URL, request body, API key, or backend error in server logs.
        pass

    def _session(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            token = cookie["archivum_session"].value if "archivum_session" in cookie else ""
        except Exception:
            token = ""
        now = time.time()
        with _sessions_lock:
            for old, state in list(_sessions.items()):
                if now - state.touched > SESSION_TTL:
                    del _sessions[old]
            if token not in _sessions:
                token = secrets.token_urlsafe(32)
                _sessions[token] = Session()
                fresh = True
            else:
                fresh = False
            state = _sessions[token]
            state.touched = now
        return token, state, fresh

    def _json(self, status, payload, token=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if token:
            cookie = f"archivum_session={token}; HttpOnly; SameSite=Strict; Path=/"
            if self.headers.get("X-Forwarded-Proto") == "https":
                cookie += "; Secure"
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in {"/api/session/key", "/api/chat", "/api/chat/reset"}:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        origin = self.headers.get("Origin")
        if origin and origin not in {
            f"http://{self.headers.get('Host')}",
            f"https://{self.headers.get('Host')}",
        }:
            return self._json(HTTPStatus.FORBIDDEN, {"error": "Invalid origin"})
        token, session, fresh = self._session()
        size = self.headers.get("Content-Length", "0")
        if not size.isdigit() or int(size) > MAX_BODY:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Invalid request size"}, token if fresh else None)
        try:
            data = json.loads(self.rfile.read(int(size))) if int(size) else {}
            if not isinstance(data, dict):
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"}, token if fresh else None)

        if self.path == "/api/session/key":
            key = data.get("api_key")
            if not isinstance(key, str) or not key.strip() or len(key) > 4096:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid API key"}, token if fresh else None)
            with session.lock:
                session.api_key = key.strip()
            return self._json(HTTPStatus.OK, {"status": "ok"}, token if fresh else None)

        if self.path == "/api/chat/reset":
            with session.lock:
                session.history = []
                session.conversation_id = secrets.token_urlsafe(18)
                conversation_id = session.conversation_id
            return self._json(HTTPStatus.OK, {"status": "ok", "conversation_id": conversation_id}, token if fresh else None)

        question = data.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid question"}, token if fresh else None)
        with session.lock:
            if data.get("conversation_id") not in (None, session.conversation_id):
                return self._json(HTTPStatus.CONFLICT, {"error": "Conversation changed. Start a new chat."}, token if fresh else None)
            try:
                from src.rag_backend import ask_rag
                result = ask_rag(question.strip(), history=session.history, api_key=session.api_key)
                if not isinstance(result, dict):
                    raise ValueError("Invalid backend output")
                if result.get("error") or result.get("status") in ("error", "failed"):
                    return self._json(HTTPStatus.BAD_GATEWAY, {"status": "error", "error": "The request could not be completed."}, token if fresh else None)
                if not isinstance(result.get("answer"), str) or not isinstance(result.get("history"), list):
                    raise ValueError("Invalid backend output")
                session.history = result["history"]
                payload = {
                    "status": result.get("status", "ok"),
                    "answer": result["answer"],
                    "sources": _source_list(result.get("sources")),
                    "citations": result.get("citations") if isinstance(result.get("citations"), (list, dict)) else [],
                    "conversation_id": session.conversation_id,
                }
                return self._json(HTTPStatus.OK, payload, token if fresh else None)
            except Exception:
                # Never echo backend exceptions; provider errors can include credentials.
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "error": "The request could not be completed."}, token if fresh else None)


if __name__ == "__main__":
    host = os.environ.get("ARCHIVUM_HOST", "127.0.0.1")
    port = int(os.environ.get("ARCHIVUM_PORT", "8000"))
    print(f"Archivum is available at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
