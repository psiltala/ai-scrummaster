"""AI Scrummaster, Azure DevOps Boards agent (stdlib only).

An A2A message/send on POST / carries the task the event dispatcher built: a work
item id and a connection id. The agent reaches Azure DevOps only through the
ado-boards MCP tools behind the gateway; it holds no credential and has no
inbound webhook route.
"""

import json
import logging
import os
import re
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.boards.mcp_client import McpClient, McpError
from src.boards.story import PlanError, ScopeError, process_story

log = logging.getLogger("scrummaster-boards")
MAX_BODY = 1 << 20
PROPAGATED_HEADERS = ("x-fabric-consumer-context", "x-request-id", "traceparent")
CONNECTION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
ID_KEYS = ("workItemId", "work_item_id", "id")
ID_TEXT_RE = re.compile(
    r"(?:work[ _-]?item(?:[ _-]?id)?|story)\D{0,8}(\d{1,12})", re.IGNORECASE
)
CONN_TEXT_RE = re.compile(
    r"connection[ _-]?id\W{0,4}([a-z0-9][a-z0-9-]{0,62})", re.IGNORECASE
)


def parse_task(text):
    """Return (work_item_id, connection_id or None) from the A2A task text, or (None, None)."""
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            wid = next((obj[k] for k in ID_KEYS if k in obj), None)
            if isinstance(wid, str) and wid.isdigit():
                wid = int(wid)
            conn = obj.get("connectionId")
            outside = {
                int(x) for x in ID_TEXT_RE.findall(text[:start] + text[end + 1 :])
            }
            if (
                isinstance(wid, int)
                and not isinstance(wid, bool)
                and wid >= 1
                and outside <= {wid}
            ):
                return wid, conn if isinstance(conn, str) and CONNECTION_RE.match(
                    conn
                ) else None
            return None, None
    # Two different ids (e.g. event text appended to the instruction) is ambiguous: refuse.
    ids = {int(x) for x in ID_TEXT_RE.findall(text)}
    conns = {x.lower() for x in CONN_TEXT_RE.findall(text)}
    if len(ids) != 1:
        return None, None
    wid = ids.pop()
    return (wid if wid >= 1 else None), (conns.pop() if len(conns) == 1 else None)


def call_llm(system, user, headers):
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    body = {
        "model": os.environ.get("OPENAI_MODEL", "fabric-reasoning"),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
            **headers,
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def handle_task(text, headers, llm=None, mcp_factory=McpClient):
    """Returns (ok, result-or-error dict)."""
    llm = llm or call_llm
    wid, conn = parse_task(text)
    if wid is None:
        return False, {"error": "task names no work item id"}
    mcp = mcp_factory(os.environ["MCP_GATEWAY_URL"].rstrip("/") + "/mcp", headers)
    try:
        out = process_story(wid, mcp, lambda s, u: llm(s, u, headers))
    except PlanError as e:
        return False, {"error": f"model plan rejected, nothing written: {e}"}
    except (McpError, ScopeError) as e:
        return False, {"error": str(e)}
    except Exception as e:  # noqa: BLE001  model/transport faults: answer, never drop the connection
        log.error("task failed: %s", type(e).__name__)
        return False, {
            "error": f"internal error ({type(e).__name__}), nothing further written"
        }
    if conn:
        out["connection"] = conn
    return True, out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/":
            return self._send(404, {"error": "not found"})
        if not self.headers.get("x-fabric-consumer-context"):
            return self._send(401, {"error": "missing consumer context"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            return self._send(
                413, {"error": "body length missing, invalid or too large"}
            )
        try:
            body = json.loads(self.rfile.read(n) or b"null")
            rid = body.get("id")
            if body.get("method") != "message/send":
                return self._send(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {"code": -32601, "message": "only message/send"},
                    },
                )
            parts = body["params"]["message"]["parts"]
            text = "\n".join(p["text"] for p in parts if p.get("kind") == "text")
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            return self._send(400, {"error": "unreadable task"})
        headers = {
            h: self.headers[h] for h in PROPAGATED_HEADERS if self.headers.get(h)
        }
        ok, out = handle_task(text, headers)
        if not ok:
            return self._send(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32000, "message": out["error"]},
                },
            )
        self._send(
            200,
            {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "kind": "task",
                    "id": str(uuid.uuid4()),
                    "contextId": str(uuid.uuid4()),
                    "status": {"state": "completed"},
                    "artifacts": [
                        {
                            "artifactId": str(uuid.uuid4()),
                            "parts": [{"kind": "text", "text": json.dumps(out)}],
                        }
                    ],
                },
            },
        )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ThreadingHTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler
    ).serve_forever()


if __name__ == "__main__":
    main()
