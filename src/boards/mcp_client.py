"""Minimal streamable-HTTP MCP client (stdlib only) for the ado-boards tools.

The agent holds no Azure DevOps credential: every call goes through the gateway,
which scopes it to the connection named by the forwarded consumer context.
"""

import json
import urllib.error
import urllib.request

TOOL_PREFIX = "ado-boards_"
TOOLS = (
    "get_work_item",
    "list_children",
    "update_story",
    "create_child_task",
    "add_comment",
    # Optional: the agent falls back to the old flow when these are off or unlisted.
    "set_story_title",
    "search_stories",
    "mark_duplicate",
)


class McpError(Exception):
    """Transport failure, JSON-RPC error, or a tool refusal (isError)."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def _rpc_payload(raw, rid):
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None
    if text[0] in "{[":
        return json.loads(text)
    best = None
    for block in text.split("\n\n"):
        data = "\n".join(
            ln[5:].lstrip(" ") for ln in block.splitlines() if ln.startswith("data:")
        )
        if not data:
            continue
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("id") == rid:
            return obj
        best = best or obj
    return best


class McpClient:
    def __init__(self, url, headers=None, timeout=60):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._session = None
        self._next = 0
        self._ready = False

    def _post(self, body, want_response):
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        for k, v in self.headers.items():
            req.add_header(k, v)
        if self._session:
            req.add_header("Mcp-Session-Id", self._session)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self._session = resp.headers.get("Mcp-Session-Id") or self._session
                raw = resp.read(1 << 20)
        except urllib.error.HTTPError as e:
            raise McpError(
                f"mcp {body.get('method')} HTTP {e.code}", f"http:{e.code}"
            ) from None
        except (urllib.error.URLError, OSError) as e:
            raise McpError(
                f"mcp {body.get('method')} unreachable: {type(e).__name__}",
                "unreachable",
            ) from None
        if not want_response:
            return None
        try:
            obj = _rpc_payload(raw, body["id"])
        except ValueError:
            raise McpError(f"mcp {body.get('method')} reply is not JSON") from None
        if not isinstance(obj, dict):
            raise McpError(f"mcp {body.get('method')} returned no response")
        if obj.get("error"):
            code = obj["error"].get("code")
            raise McpError(f"mcp {body.get('method')} rpc error {code}", f"rpc:{code}")
        return obj.get("result") or {}

    def _call(self, method, params):
        self._next += 1
        return self._post(
            {"jsonrpc": "2.0", "id": self._next, "method": method, "params": params},
            True,
        )

    def _init(self):
        if self._ready:
            return
        self._call(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ai-scrummaster-boards", "version": "1"},
            },
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, False)
        self._ready = True

    def call_tool(self, tool, args):
        """Call one ado-boards tool; return its JSON text body as a dict."""
        if tool not in TOOLS:
            raise McpError(f"tool {tool} is not an ado-boards tool", "unknown_tool")
        self._init()
        res = self._call("tools/call", {"name": TOOL_PREFIX + tool, "arguments": args})
        text = "".join(
            c.get("text", "")
            for c in res.get("content") or []
            if c.get("type") == "text"
        )
        try:
            body = json.loads(text) if text else {}
        except ValueError:
            raise McpError(f"{tool} returned non-JSON content") from None
        if res.get("isError"):
            raise McpError(
                f"{tool} refused: {body.get('code')}: {body.get('message')}",
                body.get("code"),
            )
        return body
