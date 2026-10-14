import json
import logging
import re
from collections import deque
from crewai import Crew, LLM, Process
from src.config import settings
from src.llm_json import extract_json, find_balanced_json
from src.sanitizer import SanitizedPayload
from src.schemas.triage import TriageResult, RouteType
from src.schemas.dedup import DedupResult
from src.schemas.findings import Finding
from src.schemas.story import DecomposedStories
from src.agents.story_decomposer import (
    build_story_decomposer_agent,
    build_story_decomposer_task,
)
import dataclasses
from src.tools.github_api import (
    create_issue,
    fetch_open_issue_titles,
    link_sub_issue,
    fetch_pr_diff,
    fetch_open_issues_with_dates,
    fetch_recent_activity,
    fetch_changed_dep_files,
    fetch_repo_dep_files,
)
from src.checks.staleness import find_stale_issues, check_wip_limits
from src.agents.maintenance import build_maintenance_agent, build_maintenance_task
from src.agents.standup import build_standup_agent, build_standup_task
from src.checks.secrets import scan_for_secrets
from src.checks.dependencies import extract_packages, lookup_cves_batch
from src.checks.owasp import classify_owasp
from src.checks.policy import PolicyEngine
from src.tools.github_mcp import mcp_tools_for
from src.agents.triage import build_triage_agent, build_triage_task
from src.agents.dedup import build_dedup_agent, build_dedup_task
from src.agents.devsecops import build_devsecops_agent, build_devsecops_task
from src.agents.action import build_action_agent, build_action_task

logger = logging.getLogger(__name__)

_policy_engine: PolicyEngine | None = None


def _get_policy() -> PolicyEngine:
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine(settings.policies_path)
    return _policy_engine


def _reset_policy_cache() -> None:
    """Reset the cached PolicyEngine instance. Used by tests to ensure isolation."""
    global _policy_engine
    _policy_engine = None


# Moved to src/llm_json so the Boards agent can share it without importing crewai.
_find_balanced_json = find_balanced_json
_extract_json = extract_json


def _make_llm() -> LLM:
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
    )


def _parse_triage(output: str) -> TriageResult:
    try:
        raw = json.loads(_extract_json(output))
        return TriageResult.model_validate(raw)
    except Exception:
        logger.warning(
            "Triage parse failed, defaulting to issue route: %s", output[:200]
        )
        return TriageResult(route=RouteType.issue, repo="unknown", entity_id=None)


def _parse_dedup(output: str) -> DedupResult:
    try:
        raw = json.loads(_extract_json(output))
        return DedupResult.model_validate(raw)
    except Exception:
        return DedupResult(is_duplicate=False)


def _parse_findings(output: str) -> list[Finding]:
    try:
        raw = json.loads(_extract_json(output))
        items = (
            raw
            if isinstance(raw, list)
            else raw.get("findings", [])
            if isinstance(raw, dict)
            else []
        )
        return [Finding.model_validate(f) for f in items]
    except Exception:
        logger.warning("Findings parse failed: %s", output[:200])
    return []


def _repair_json(s: str) -> str:
    """Escape literal newlines/tabs inside JSON string values (common local-LLM failure)."""
    result = []
    in_string = False
    escaped = False
    for ch in s:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _parse_decomposed(output: str) -> DecomposedStories:
    extracted = _extract_json(output)
    for candidate in (extracted, _repair_json(extracted)):
        try:
            raw = json.loads(candidate)
            if isinstance(raw, list):
                raw = {"tickets": raw}
            return DecomposedStories.model_validate(raw)
        except Exception:
            continue
    logger.warning("Decompose parse failed: %s", output[:200])
    return DecomposedStories(tickets=[])


def _topo_sort(tickets: list) -> list[int]:
    """Returns ticket indices in topological order (dependencies before dependents)."""
    n = len(tickets)
    indegree = [0] * n
    graph: list[list[int]] = [[] for _ in range(n)]
    for i, ticket in enumerate(tickets):
        for dep in ticket.depends_on:
            if 0 <= dep < n:
                graph[dep].append(i)
                indegree[i] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    # Cycle protection: append any indices not yet processed
    seen = set(order)
    remaining = [i for i in range(n) if i not in seen]
    if remaining:
        logger.warning(
            "Dependency cycle detected among ticket indices %s; appending in index order",
            remaining,
        )
        order.extend(remaining)
    return order


_STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "for",
    "and",
    "or",
    "with",
    "is",
    "be",
    "on",
}
_DEDUP_THRESHOLD = 0.7


def _title_is_duplicate(title: str, existing: list[str]) -> bool:
    """True if title overlaps sufficiently with any existing issue title (Jaccard >= threshold)."""

    def tokens(s: str) -> set[str]:
        return {
            w.lower() for w in re.split(r"\W+", s) if w and w.lower() not in _STOP_WORDS
        }

    new_tokens = tokens(title)
    if not new_tokens:
        return False
    for existing_title in existing:
        ex_tokens = tokens(existing_title)
        if not ex_tokens:
            continue
        if (
            len(new_tokens & ex_tokens) / len(new_tokens | ex_tokens)
            >= _DEDUP_THRESHOLD
        ):
            return True
    return False


async def run_crew_for_event(payload: SanitizedPayload, raw_event: dict) -> dict:
    """Main entry point: sanitized payload -> crew run -> dict summary of actions taken."""
    llm = _make_llm()
    policy = _get_policy()

    # --- Triage ---
    # Skip LLM triage for story and scan events — route and repo are already known.
    if raw_event.get("event_type") == "story":
        triage = TriageResult(route=RouteType.story, repo=payload.repo, entity_id=None)
    elif raw_event.get("event_type") == "scan":
        triage = TriageResult(route=RouteType.scan, repo=payload.repo, entity_id=None)
    else:
        with mcp_tools_for("triage") as _triage_tools:
            triage_agent = build_triage_agent(llm)
            triage_task = build_triage_task(
                triage_agent,
                {
                    "event_type": raw_event.get("event_type", "issues"),
                    "action": payload.action,
                    "repo": payload.repo,
                    "entity_id": payload.entity_id,
                    "title": payload.title,
                },
            )
            crew = Crew(
                agents=[triage_agent], tasks=[triage_task], process=Process.sequential
            )
            triage_output = await crew.kickoff_async()
        triage = _parse_triage(str(triage_output))
    logger.info("Triage result: %s", triage.model_dump())

    # --- Fetch PR diff ---
    if triage.route == RouteType.pr and triage.entity_id:
        from src.sanitizer import sanitize_field

        diff = await fetch_pr_diff(settings.github_token, triage.repo, triage.entity_id)
        if diff:
            payload = dataclasses.replace(payload, diff=sanitize_field(diff, "diff"))

    # --- Fetch dependency files for push events ---
    if raw_event.get("event_type") == "push" and raw_event.get("commits"):
        ref = raw_event.get("ref", "").replace("refs/heads/", "")
        if ref:
            dep_files_fetched = await fetch_changed_dep_files(
                settings.github_token, triage.repo, ref, raw_event["commits"]
            )
            raw_event["changed_files"] = dep_files_fetched

    # --- Fetch repo dep files for scan events ---
    if triage.route == RouteType.scan:
        raw_event["changed_files"] = await fetch_repo_dep_files(
            settings.github_token, triage.repo
        )

    # --- Pre-scan (deterministic, no LLM) ---
    secret_findings = scan_for_secrets(payload.diff) if payload.diff else []
    owasp_categories = classify_owasp(f"{payload.title} {payload.body}")

    cve_findings: list[Finding] = []
    dep_files = raw_event.get("changed_files", {})
    for filename, content in dep_files.items():
        packages = extract_packages(content, filename)
        if packages:
            cve_findings.extend(await lookup_cves_batch(packages))

    # --- Scan events return findings directly (no entity to comment on) ---
    if triage.route == RouteType.scan:
        all_findings = secret_findings + cve_findings
        return {
            "status": "scan_complete",
            "repo": triage.repo,
            "findings": len(all_findings),
            "cves": len(cve_findings),
        }

    policy_findings: list[Finding] = []
    if triage.route.value == "pr":
        branch = raw_event.get("pull_request", {}).get("head", {}).get("ref", "")
        if branch and not policy.check_branch_name(branch):
            from src.schemas.findings import FindingType, Severity

            policy_findings.append(
                Finding(
                    type=FindingType.policy,
                    severity=Severity.medium,
                    description=f"Branch '{branch}' does not follow naming convention: {policy.rules.branch_naming.pattern}",
                    recommendation=f"Rename branch to match pattern: {policy.rules.branch_naming.pattern}",
                )
            )

    # --- Story Decomposer (only for story events) ---
    if triage.route == RouteType.story:
        decomposer_agent = build_story_decomposer_agent(llm)
        decomposer_task = build_story_decomposer_task(
            decomposer_agent, triage.repo, raw_event.get("story", "")
        )
        crew = Crew(
            agents=[decomposer_agent],
            tasks=[decomposer_task],
            process=Process.sequential,
        )
        decomposer_output = await crew.kickoff_async()

        decomposed = _parse_decomposed(str(decomposer_output))
        order = _topo_sort(decomposed.tickets)

        existing_titles = await fetch_open_issue_titles(
            settings.github_token, triage.repo
        )

        index_to_number: dict[int, int] = {}
        index_to_id: dict[int, int] = {}
        skipped = 0
        for idx in order:
            ticket = decomposed.tickets[idx]
            if _title_is_duplicate(ticket.title, existing_titles):
                logger.info("Skipping duplicate ticket: %s", ticket.title)
                skipped += 1
                continue
            labels = [
                f"type:{ticket.type}",
                f"category:{ticket.category}",
                f"size:{ticket.size}",
                f"priority:{ticket.priority}",
            ]
            number, db_id = await create_issue(
                token=settings.github_token,
                repo=triage.repo,
                title=ticket.title,
                body=ticket.body,
                labels=labels,
            )
            index_to_number[idx] = number
            index_to_id[idx] = db_id
            existing_titles.append(ticket.title)
            logger.info("Created issue #%d: %s", number, ticket.title)

        for idx, ticket in enumerate(decomposed.tickets):
            if idx not in index_to_id:
                continue  # ticket was skipped as duplicate
            if ticket.type in ("story", "task"):
                for dep_idx in ticket.depends_on:
                    if (
                        dep_idx in index_to_number
                        and decomposed.tickets[dep_idx].type == "epic"
                    ):
                        try:
                            await link_sub_issue(
                                token=settings.github_token,
                                repo=triage.repo,
                                parent_number=index_to_number[dep_idx],
                                child_id=index_to_id[idx],
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to link issue #%d as sub-issue of #%d: %s",
                                index_to_number[idx],
                                index_to_number[dep_idx],
                                exc,
                            )

        return {
            "status": "story_created",
            "repo": triage.repo,
            "tickets_created": len(index_to_number),
            "tickets_skipped": skipped,
        }

    # --- Dedup (only for issues) ---
    dedup = DedupResult(is_duplicate=False)
    if triage.route == RouteType.issue and triage.entity_id:
        with mcp_tools_for("dedup") as dedup_tools:
            dedup_agent = build_dedup_agent(llm, dedup_tools)
            dedup_task = build_dedup_task(
                dedup_agent,
                triage.repo,
                triage.entity_id,
                payload.title,
                payload.body,
                policy.rules.dedup_confidence_threshold,
            )
            crew = Crew(
                agents=[dedup_agent], tasks=[dedup_task], process=Process.sequential
            )
            dedup_output = await crew.kickoff_async()
        dedup = _parse_dedup(str(dedup_output))

    # --- DevSecOps ---
    all_pre_findings = secret_findings + cve_findings + policy_findings
    with mcp_tools_for("devsecops") as ds_tools:
        ds_agent = build_devsecops_agent(llm, ds_tools)
        ds_task = build_devsecops_task(
            agent=ds_agent,
            event_type=triage.route.value,
            repo=triage.repo,
            entity_id=triage.entity_id or 0,
            title=payload.title,
            body=payload.body,
            diff=payload.diff,
            owasp_categories=owasp_categories,
            secret_findings=secret_findings,
            cve_findings=cve_findings,
        )
        crew = Crew(agents=[ds_agent], tasks=[ds_task], process=Process.sequential)
        ds_output = await crew.kickoff_async()

    llm_findings = _parse_findings(str(ds_output))
    all_findings = all_pre_findings + llm_findings

    # --- Action ---
    pr_number = triage.entity_id if triage.route == RouteType.pr else None
    with mcp_tools_for("action") as action_tools:
        action_agent = build_action_agent(llm, action_tools)
        action_task = build_action_task(
            agent=action_agent,
            triage=triage,
            findings=all_findings,
            dedup=dedup,
            pr_number=pr_number,
            repo=triage.repo,
            dedup_confidence_threshold=policy.rules.dedup_confidence_threshold,
            pr_author=payload.pr_author,
        )
        crew = Crew(
            agents=[action_agent], tasks=[action_task], process=Process.sequential
        )
        action_output = await crew.kickoff_async()

    return {
        "status": "processed",
        "route": triage.route.value,
        "findings": len(all_findings),
        "is_duplicate": dedup.is_duplicate,
        "action_summary": str(action_output)[:500],
    }


async def run_maintenance(repo: str) -> dict:
    """Run stale issue detection and WIP limit enforcement."""
    llm = _make_llm()
    policy = _get_policy()

    issues = await fetch_open_issues_with_dates(settings.github_token, repo)

    stale = find_stale_issues(issues, policy.rules.stale_days)
    wip_violations = check_wip_limits(issues, policy.rules.wip_limits)

    from datetime import datetime, timezone

    stale_nudge = []
    stale_close = []
    for issue in stale:
        updated = datetime.fromisoformat(issue["updated_at"])
        days = (datetime.now(timezone.utc) - updated).days
        entry = {**issue, "days_stale": days}
        if days >= policy.rules.auto_close_days:
            stale_close.append(entry)
        else:
            stale_nudge.append(entry)

    if not stale_nudge and not stale_close and not wip_violations:
        return {"status": "clean", "repo": repo}

    with mcp_tools_for("action") as tools:
        agent = build_maintenance_agent(llm, tools)
        task = build_maintenance_task(
            agent=agent,
            stale_issues=stale_nudge,
            auto_close_issues=stale_close,
            wip_violations=wip_violations,
            repo=repo,
            stale_nudge_message=policy.rules.stale_nudge_message,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        await crew.kickoff_async()

    return {
        "status": "maintenance_complete",
        "repo": repo,
        "stale_nudged": len(stale_nudge),
        "stale_closed": len(stale_close),
        "wip_violations": len(wip_violations),
    }


async def run_standup(repo: str, since_hours: int = 24) -> dict:
    """Generate and post a daily standup summary."""
    llm = _make_llm()

    activity = await fetch_recent_activity(settings.github_token, repo, since_hours)

    agent = build_standup_agent(llm)
    task = build_standup_task(agent, activity, repo)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    output = await crew.kickoff_async()

    summary = str(output)
    from datetime import date

    title = f"Daily Standup - {date.today().isoformat()}"

    await create_issue(
        token=settings.github_token,
        repo=repo,
        title=title,
        body=summary,
        labels=["standup"],
    )

    return {"status": "standup_posted", "repo": repo, "title": title}
