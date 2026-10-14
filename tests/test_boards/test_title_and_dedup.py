"""Clear titles and duplicate handling, against the fake ado-boards MCP server."""

import json

import pytest

from src.boards import story as boards
from src.boards.mcp_client import McpClient
from tests.test_boards import test_boards_agent as base

PLAN = base.PLAN
fake = base.fake  # the fake ado-boards MCP server fixture

BOTH = boards.Flags(title=True, dedup=True)
WRITES = (
    "update_story",
    "create_child_task",
    "add_comment",
    "set_story_title",
    "mark_duplicate",
)


def scripted(plan=PLAN, queries=("export report",), matches=(), covered=(), seen=None):
    """A fake model that answers each prompt kind with a fixed JSON object."""

    def llm(system, user):
        if seen is not None:
            seen.append((system, user))
        if system == boards.DEDUP_QUERY_PROMPT:
            return json.dumps({"queries": list(queries)})
        if system == boards.DEDUP_JUDGE_PROMPT:
            return json.dumps({"matches": list(matches)})
        if system == boards.TASK_COVER_PROMPT:
            return json.dumps({"covered": list(covered)})
        assert system.startswith(boards.SYSTEM_PROMPT), system
        return json.dumps(plan)

    return llm


def titled(new_title=None):
    plan = json.loads(json.dumps(PLAN))
    plan["title_is_clear"] = new_title is None
    plan["new_title"] = new_title or "Export"
    return plan


def writes(fake):
    return [n for n, _ in fake.calls if n in WRITES]


def comment(fake):
    texts = [a["comment"] for n, a in fake.calls if n == "add_comment"]
    assert len(texts) == 1
    return texts[0]


def row(wid, title, state="New", text=""):
    return {
        "id": wid,
        "title": title,
        "state": state,
        "priority": 2,
        "createdDate": "2026-10-01T00:00:00Z",
        "descriptionText": text,
    }


# --- Duplicate story ---


def test_clear_duplicate_only_marks_it_and_stops(fake):
    fake.corpus = {
        60: row(60, "Export the sales report as CSV"),
        80: row(80, "CSV export of sales report"),
    }
    seen = []
    match = [
        {"id": 60, "same_intent": True, "confidence": "high", "reason": "Same export."},
        {"id": 80, "same_intent": True, "confidence": "high", "reason": "Same."},
    ]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(matches=match, seen=seen), BOTH
    )
    assert out["status"] == "duplicate" and out["duplicate_of"] == 60 and out["linked"]
    assert writes(fake) == ["mark_duplicate"]
    args = next(a for n, a in fake.calls if n == "mark_duplicate")
    assert args["storyId"] == 100 and args["duplicateOfId"] == 60
    assert "#60" in args["comment"] and "Same export." in args["comment"]
    assert "Nothing was closed or deleted" in args["comment"]
    assert [s for s, _ in seen] == [
        boards.DEDUP_QUERY_PROMPT,
        boards.DEDUP_JUDGE_PROMPT,
    ]  # no plan, no enhancement


def test_near_but_different_story_runs_the_normal_flow(fake):
    fake.corpus = {60: row(60, "Export the sales report as PDF with charts")}
    match = [
        {
            "id": 60,
            "same_intent": False,
            "confidence": "high",
            "reason": "PDF, not CSV.",
        }
    ]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), matches=match), BOTH
    )
    assert out["status"] == "processed" and len(out["tasks_created"]) == 3
    assert "mark_duplicate" not in writes(fake)
    assert "Duplicate check: no clear duplicate among 1 similar" in comment(fake)


def test_medium_confidence_is_not_a_duplicate(fake):
    fake.corpus = {60: row(60, "Export report")}
    match = [{"id": 60, "same_intent": True, "confidence": "medium", "reason": "?"}]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), matches=match), BOTH
    )
    assert out["status"] == "processed" and "mark_duplicate" not in writes(fake)


def test_self_children_newer_and_closed_stories_are_never_candidates(fake):
    _add_child(fake, 70, "Export report child")
    fake.corpus = {
        70: row(70, "Export report child"),
        120: row(120, "Export report again"),
        50: row(50, "Export report old", state="Closed"),
    }
    seen = []
    match = [
        {"id": i, "same_intent": True, "confidence": "high", "reason": "x"}
        for i in (100, 70, 120, 50)
    ]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), matches=match, seen=seen), BOTH
    )
    assert out["status"] == "processed" and "mark_duplicate" not in writes(fake)
    # Nothing left to judge, so the judge is never asked.
    assert boards.DEDUP_JUDGE_PROMPT not in [s for s, _ in seen]
    assert "no clear duplicate among 0 similar" in comment(fake)


def test_a_judged_id_that_search_did_not_return_is_ignored(fake):
    fake.corpus = {60: row(60, "Export report")}
    match = [{"id": 7, "same_intent": True, "confidence": "high", "reason": "x"}]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), matches=match), BOTH
    )
    assert out["status"] == "processed" and "mark_duplicate" not in writes(fake)


def test_scope_guard_refuses_mark_duplicate_on_an_unsearched_story(fake):
    scoped = boards.ScopedBoards(McpClient(fake.url), 100)
    with pytest.raises(boards.ScopeError):
        scoped.mark_duplicate(60, "dup")
    assert fake.calls == []


def test_unreadable_judge_answer_is_no_duplicate(fake):
    fake.corpus = {60: row(60, "Export report")}
    base = scripted(titled())

    def llm(system, user):
        return "no json" if system == boards.DEDUP_JUDGE_PROMPT else base(system, user)

    out = boards.process_story(100, McpClient(fake.url), llm, BOTH)
    assert out["status"] == "processed" and "mark_duplicate" not in writes(fake)


# --- Title ---


def test_unclear_title_is_rewritten_and_the_original_kept(fake):
    new = "Export the monthly report as CSV from the reports page"
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(new)), boards.Flags(title=True)
    )
    assert out["status"] == "processed" and out["title_changed"]
    assert fake.items[100]["title"] == new
    sets = [a for n, a in fake.calls if n == "set_story_title"]
    assert sets == [{"storyId": 100, "title": new}]
    assert "<p><b>Original title:</b> Export</p>" in fake.items[100]["description"]
    # The rewrite (with the original title) is written before the title changes.
    assert writes(fake).index("update_story") < writes(fake).index("set_story_title")
    assert f'Changed the title from "Export" to "{new}"' in comment(fake)


def test_clear_title_is_left_alone(fake):
    fake.items[100]["title"] = "Export the monthly report as CSV from the reports page"
    seen = []
    out = boards.process_story(
        100,
        McpClient(fake.url),
        scripted(titled(), seen=seen),
        boards.Flags(title=True),
    )
    assert out["status"] == "processed" and not out["title_changed"]
    assert "set_story_title" not in writes(fake)
    assert "Original title" not in fake.items[100]["description"]
    assert "Title kept: it already says what the story is." in comment(fake)
    assert "title_is_clear" in seen[0][0]


def test_same_title_answer_makes_no_call(fake):
    out = boards.process_story(
        100,
        McpClient(fake.url),
        scripted(titled("  export ")),
        boards.Flags(title=True),
    )
    assert not out["title_changed"] and "set_story_title" not in writes(fake)


def test_title_flag_off_keeps_the_old_prompt_and_no_title_call(fake):
    seen = []
    boards.process_story(
        100, McpClient(fake.url), scripted(titled("A much better title"), seen=seen)
    )
    assert seen[0][0] == boards.SYSTEM_PROMPT
    assert "set_story_title" not in writes(fake)
    assert "Title" not in comment(fake)


def test_resume_applies_a_title_change_that_a_crash_skipped(fake):
    new = "Export the monthly report as CSV"
    fake.off = {"set_story_title"}
    boards.process_story(
        100, McpClient(fake.url), scripted(titled(new)), boards.Flags(title=True)
    )
    assert fake.items[100]["title"] == "Export"
    fake.off = set()
    fake.calls.clear()

    def no_llm(*_):
        raise AssertionError("the model must not be called")

    out = boards.process_story(
        100, McpClient(fake.url), no_llm, boards.Flags(title=True)
    )
    assert out["status"] == "processed" and out["title_changed"] and not out["enhanced"]
    assert fake.items[100]["title"] == new
    assert writes(fake) == ["set_story_title", "add_comment"]


# --- Platform tools off or unlisted: the old behaviour ---


@pytest.mark.parametrize("mode", ["off", "unlisted"])
def test_new_tools_unavailable_falls_back_to_the_old_flow(fake, mode):
    tools = {"search_stories", "set_story_title", "mark_duplicate"}
    setattr(fake, mode, tools)
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled("Export the report as CSV")), BOTH
    )
    assert out["status"] == "processed" and out["enhanced"]
    assert len(out["tasks_created"]) == 3 and not out["title_changed"]
    text = comment(fake)
    code = "flag_off" if mode == "off" else "unknown_tool"
    assert f"Duplicate check skipped: search_stories is unavailable ({code})" in text
    assert f"Title not changed: set_story_title is unavailable ({code})" in text


def test_mark_duplicate_off_handles_the_story_as_new(fake):
    fake.corpus = {60: row(60, "Export report")}
    fake.off = {"mark_duplicate"}
    match = [{"id": 60, "same_intent": True, "confidence": "high", "reason": "x"}]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), matches=match), BOTH
    )
    assert out["status"] == "processed" and len(out["tasks_created"]) == 3
    assert "#60 looks like the same story, but mark_duplicate is unavailable" in (
        comment(fake)
    )


# --- Child tasks ---


def _add_child(fake, cid, title):
    fake.items[cid] = {
        "id": cid,
        "type": "Task",
        "title": title,
        "state": "To Do",
        "priority": 2,
    }
    fake.items[100]["childIds"].append(cid)


def test_existing_child_covering_a_planned_task_is_skipped_and_noted(fake):
    _add_child(fake, 400, "Implement CSV export API")
    seen = []
    cover = [{"task": 1, "by": 400, "reason": "Same endpoint."}]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), covered=cover, seen=seen), BOTH
    )
    made = [a["title"] for n, a in fake.calls if n == "create_child_task"]
    assert made == ["Add export button", "Write export tests"]
    assert out["tasks_skipped"] == ["Add export endpoint"]
    assert (
        'Skipped task "Add export endpoint": Task #400 "Implement CSV export API" '
        "already covers it (Same endpoint.)." in comment(fake)
    )
    cover_prompt = next(u for s, u in seen if s == boards.TASK_COVER_PROMPT)
    assert "id 400: Implement CSV export API" in cover_prompt


def test_covered_answer_naming_an_unknown_child_creates_the_task(fake):
    _add_child(fake, 400, "Something else")
    cover = [{"task": 1, "by": 999, "reason": "x"}, {"task": 9, "by": 400}]
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), covered=cover), BOTH
    )
    assert len(out["tasks_created"]) == 3


def test_no_overlap_question_without_foreign_children(fake):
    seen = []
    boards.process_story(100, McpClient(fake.url), scripted(titled(), seen=seen), BOTH)
    assert boards.TASK_COVER_PROMPT not in [s for s, _ in seen]


def test_dedup_flag_off_keeps_title_only_task_check(fake):
    _add_child(fake, 400, "Implement CSV export API")
    seen = []
    out = boards.process_story(
        100, McpClient(fake.url), scripted(titled(), seen=seen), boards.Flags()
    )
    assert len(out["tasks_created"]) == 3
    assert [s for s, _ in seen] == [boards.SYSTEM_PROMPT]
    assert not [n for n, _ in fake.calls if n == "search_stories"]


# --- Flags ---


def test_flags_default_off_and_read_env(monkeypatch):
    monkeypatch.delenv("SCRUMMASTER_TITLE_REWRITE_ENABLED", raising=False)
    monkeypatch.delenv("SCRUMMASTER_DEDUP_ENABLED", raising=False)
    assert boards.Flags.from_env() == boards.Flags(title=False, dedup=False)
    monkeypatch.setenv("SCRUMMASTER_TITLE_REWRITE_ENABLED", "true")
    monkeypatch.setenv("SCRUMMASTER_DEDUP_ENABLED", "off")
    assert boards.Flags.from_env() == boards.Flags(title=True, dedup=False)


# --- Golden cases ---


def test_golden_create_gol_gets_a_clear_title_then_a_second_story_is_its_duplicate(
    fake,
):
    fake.items[100].update(title="create gol", description="<p>create gol</p>")
    gol_title = "Create a Conway's Game of Life simulation"
    out = boards.process_story(
        100,
        McpClient(fake.url),
        scripted(titled(gol_title), queries=("gol", "game of life")),
        BOTH,
    )
    assert out["status"] == "processed" and fake.items[100]["title"] == gol_title
    assert "<p><b>Original title:</b> create gol</p>" in fake.items[100]["description"]

    # A second story with the same intent arrives later (higher id).
    fake.items[101] = dict(
        fake.items[100],
        id=101,
        title="create game of life",
        description="<p>create game of life</p>",
        childIds=[],
    )
    fake.story_id = 101
    fake.calls.clear()
    seen = []

    def judge(system, user):
        seen.append((system, user))
        if system == boards.DEDUP_JUDGE_PROMPT:
            assert "id 100: title: " + gol_title in user
            return json.dumps(
                {
                    "matches": [
                        {
                            "id": 100,
                            "same_intent": True,
                            "confidence": "high",
                            "reason": "Both ask for a Game of Life simulation.",
                        }
                    ]
                }
            )
        return json.dumps({"queries": ["game of life"]})

    out = boards.process_story(101, McpClient(fake.url), judge, BOTH)
    assert out["status"] == "duplicate" and out["duplicate_of"] == 100
    assert writes(fake) == ["mark_duplicate"] and fake.items[101]["duplicateOf"] == 100
