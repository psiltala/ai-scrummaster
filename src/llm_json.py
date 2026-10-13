"""Pull a JSON block out of LLM output. Stdlib only: shared by the GitHub crew and the Boards agent."""

import re

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.DOTALL)


def find_balanced_json(output: str) -> str | None:
    """Find the first balanced JSON object or array using bracket matching."""
    openers = {"{": "}", "[": "]"}
    for i, ch in enumerate(output):
        if ch not in openers:
            continue
        closer = openers[ch]
        depth = 0
        in_string = False
        escaped = False
        for j in range(i, len(output)):
            c = output[j]
            if in_string and escaped:
                escaped = False
                continue
            if in_string and c == "\\":
                escaped = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == ch:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return output[i : j + 1]
    return None


def extract_json(output: str) -> str:
    """Extract a JSON block from LLM output that may include markdown fences or prose."""
    m = _JSON_FENCE.search(output)
    if m:
        return m.group(1).strip()
    result = find_balanced_json(output)
    if result:
        return result.strip()
    return output.strip()
