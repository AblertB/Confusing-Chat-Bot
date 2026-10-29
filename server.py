import json
import re
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chatbot import ChatbotError, generate_reply


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_REQUEST_BYTES = 16_384
MAX_HISTORY_TURNS = 10
sessions = {}
sessions_lock = threading.Lock()


class ChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "web"), **kwargs)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/reset"):
            self._send_json(404, {"error": "Endpoint not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._send_json(400, {"error": "Request must be between 1 byte and 16 KB."})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be a valid JSON object."})
            return

        if self.path == "/api/reset":
            session_id = payload.get("sessionId")
            if isinstance(session_id, str):
                with sessions_lock:
                    sessions.pop(session_id, None)
            self._send_json(200, {"ok": True})
            return

        message = payload.get("message")
        level = payload.get("level", "medium")
        session_id = payload.get("sessionId")
        if not isinstance(message, str):
            self._send_json(400, {"error": "Message must be a string."})
            return
        if not isinstance(level, str):
            self._send_json(400, {"error": "Style level must be a string."})
            return
        if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9-]{36}", session_id):
            session_id = str(uuid.uuid4())

        with sessions_lock:
            history = list(sessions.get(session_id, []))

        try:
            base_answer, reply, evaluation = generate_reply(message, level, history)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except ChatbotError as error:
            self._send_json(502, {"error": str(error)})
            return

        with sessions_lock:
            sessions[session_id] = (
                history + [{"user": message.strip(), "base": base_answer}]
            )[-MAX_HISTORY_TURNS:]
        self._send_json(200, {
            "reply": reply,
            "sessionId": session_id,
            "evaluation": evaluation,
        })

    def log_message(self, format, *args):
        super().log_message(format, *args)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), ChatHandler)
    print("Tangled Chat is available at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Tangled Chat.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
