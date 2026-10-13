# AI Scrum Master

[![CI](https://github.com/mangobanaani/ai-scrummaster/actions/workflows/ci.yml/badge.svg)](https://github.com/mangobanaani/ai-scrummaster/actions/workflows/ci.yml)
[![Release](https://github.com/mangobanaani/ai-scrummaster/actions/workflows/release.yml/badge.svg)](https://github.com/mangobanaani/ai-scrummaster/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/mangobanaani/ai-scrummaster)](https://github.com/mangobanaani/ai-scrummaster/releases)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An AI-powered scrum master that automates GitHub project management — triaging issues, decomposing stories into tickets, detecting duplicates, running security checks on pull requests, and maintaining project hygiene.

## What it does

- **Story decomposition** — submit a plain-text feature description and the crew breaks it down into epics, stories, and tasks, creates GitHub issues with labels, and links sub-issues automatically
- **Deduplication** — before creating tickets, existing open issues are checked for overlap (Jaccard similarity) so the same story can be submitted multiple times safely
- **Triage** — incoming GitHub webhook events are classified and routed to the appropriate handler
- **DevSecOps** — pull requests and issues are scanned for secrets (17 regex patterns), CVEs via the OSV API (with CVSS vector severity parsing), OWASP categories, and branch naming policy violations
- **Security scan** — on-demand repository scan fetches dependency manifests (requirements.txt, package.json, go.mod) and checks for known vulnerabilities
- **Maintenance** — detects stale issues, enforces WIP limits, and auto-closes abandoned issues based on configurable policy
- **Standup** — generates daily standup summaries from recent issue and PR activity

## Architecture

```
POST /stories      →  story decomposer agent  →  GitHub issues (with sub-issue linking)
POST /webhook      →  triage agent  →  dedup + devsecops + action agents
POST /scan         →  fetch dep files  →  CVE scan  →  findings summary
POST /maintenance  →  stale/WIP check  →  maintenance agent  →  nudge/close issues
POST /standup      →  recent activity  →  standup agent  →  summary issue
```

Agents are built with [CrewAI](https://github.com/crewai-ai/crewai) and run against a local [Ollama](https://ollama.com) instance. GitHub interactions use the REST API and an MCP server (SSE transport).

## Requirements

- Python 3.12+
- Docker and Docker Compose
- A GitHub personal access token with `repo` scope
- Ollama running `qwen2.5:27b` (production) or `qwen2.5:7b` (development)

## Setup

Copy `.env.example` to `.env` and fill in the values:

```
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=your-webhook-secret
API_KEY=your-api-key
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:27b
MCP_SERVER_URL=http://github-mcp:3000
```

All three credential fields (`GITHUB_TOKEN`, `GITHUB_WEBHOOK_SECRET`, `API_KEY`) are required — the app will fail to start if any are missing.

Start the stack:

```bash
docker compose up -d
```

## Usage

### Decompose a story into tickets

```bash
curl -X POST http://localhost:8000/stories \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{"repo":"owner/repo","story":"As a user I want to log in with email and password."}'
```

### Trigger a security scan

```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{"repo":"owner/repo"}'
```

### Run maintenance

```bash
curl -X POST http://localhost:8000/maintenance \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{"repo":"owner/repo"}'
```

### Generate a standup summary

```bash
curl -X POST http://localhost:8000/standup \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{"repo":"owner/repo","since_hours":24}'
```

The `since_hours` parameter accepts an integer between 1 and 720 (default: 24).

### GitHub webhook

Point your repository webhook at `POST /webhook` with `application/json` content type and the same secret as `GITHUB_WEBHOOK_SECRET`. Supported events: `issues`, `pull_request`, `push`.

## Azure DevOps Boards entry point

`src/boards/` is a second entry point for Azure DevOps Boards. It runs as an A2A agent on the Intelligent Fabric platform: the event dispatcher sends `message/send` on `POST /` when a User Story is created. The agent rewrites the story, adds Given/When/Then acceptance criteria, sets a priority, creates 3-8 child tasks, and comments on the story with what it changed.

- Stdlib only (no crewai). It shares `src/llm_json.py` with the GitHub crew.
- It holds no Azure DevOps credential. Every read and write goes through the `ado-boards` MCP tools behind the platform gateway.
- Env: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` (default `fabric-reasoning`), `MCP_GATEWAY_URL`, `PORT` (default `8080`).
- Health: `GET /healthz`.

```bash
python -m pytest --noconftest tests/test_boards   # no crewai needed
python -m src.boards.app                          # local run
gcloud builds submit --region=europe-north1 --config=cloudbuild-boards.yaml \
  --substitutions=SHORT_SHA=$(git rev-parse --short HEAD) .
```

The platform deployment pins the built image by digest. The manifests live in the platform repo, not here.

## Policy configuration

Edit `policies/rules.yaml` to configure:

- **WIP limits** — per-label caps on open issues (e.g., feature: 5, bug: 10)
- **Stale thresholds** — days before nudge (default: 7) and auto-close (default: 30)
- **Branch naming** — regex pattern for allowed branch names
- **CVE policy** — severity levels that trigger automatic ticket creation
- **Dedup threshold** — confidence level for LLM-based duplicate detection

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
