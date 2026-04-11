# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a workspace for analyzing [Hermes Agent](https://github.com/nousresearch/hermes-agent) v0.8.0. The actual source lives in `hermes-agent/`. The root contains reference documentation and analysis notes.

## Development Commands

```bash
cd hermes-agent

# Setup
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"

# Run tests (exclude integration tests)
pytest tests/ -q --ignore=tests/integration --ignore=tests/e2e --tb=short -n auto

# Single test file
pytest tests/tools/test_approval.py -v

# Run hermes locally
python -c "from cli import main; main()"

# Lint/typecheck website
cd website && npm install && npm run typecheck
```

## Architecture

Hermes is a layered agent system with 6 distinct layers:

```
Entry → Control (hermes_cli/main.py) → Shells (cli.py, gateway/run.py) → Core (run_agent.py AIAgent) → Capabilities (agent/, tools/, model_tools.py) → State (hermes_state.py, gateway/session.py)
```

**Entry points** (defined in `pyproject.toml:99-102`):
- `hermes` → `hermes_cli.main:main` — unified CLI
- `hermes-agent` → `run_agent:main` — agent kernel direct
- `hermes-acp` → `acp_adapter.entry:main` — ACP adapter

**Call chain to AIAgent**:
```
hermes script → hermes_cli.main:main() → cmd_chat() → cli.main() → HermesCLI.__init__() → _init_agent() → AIAgent.__init__()
```

**Core components**:
- `run_agent.py:AIAgent` (L433-1225) — central orchestrator: prompt building, model loops, tool dispatch, compression, session persistence
- `cli.py:HermesCLI` — interactive TUI with prompt_toolkit
- `model_tools.py` — tool discovery and orchestration (imports `tools/*.py` which self-register via `registry.register()`)
- `hermes_state.py:SessionDB` — SQLite with FTS5 full-text search
- `gateway/run.py:GatewayRunner` — messaging platform lifecycle

## Key Design Patterns

**Self-registering tools**: Each `tools/*.py` module calls `registry.register()` at import time. Tool discovery happens in `model_tools._discover_tools()`.

**Toolset system**: Tools are grouped into toolsets (`web`, `terminal`, `file`, `browser`, etc.) that can be enabled/disabled per platform. See `toolsets.py`.

**Session persistence**: All conversations stored in SQLite via `SessionDB`. JSON logs go to `~/.hermes/sessions/`. Two-tier write: JSON first (debug), then SQLite (searchable).

**Ephemeral injection**: `ephemeral_system_prompt` and `prefill_messages` injected at API call time only — never persisted to database or logs.

**Provider abstraction**: Works with any OpenAI-compatible API (OpenRouter, Nous Portal, custom endpoints). Provider resolution at init time.

**Context compression**: When approaching token limits, `ContextCompressor` does structured summarization with `Goal/Progress/Key Decisions/Relevant Files/Next Steps/Critical Context` handoff format.

## Important Conventions

- **Skill vs Tool**: Most capabilities should be skills (procedural memory). Tools require custom Python integration or API key management. See `CONTRIBUTING.md` section "Should it be a Skill or a Tool?".
- **Bundled skills** go in `skills/`, official optional skills in `optional-skills/`. Skills are `SKILL.md` + optional `scripts/` directory.
- **Cross-platform**: Never assume Unix. `termios`/`fcntl` are Unix-only. Use `pathlib.Path`. Handle encoding errors for Windows `.env` files.
- **Security**: Always use `shlex.quote()` when interpolating user input into shell commands. Resolve symlinks with `os.path.realpath()` before path checks.
- **Commit style**: Conventional Commits — `fix(cli):`, `feat(gateway):`, `test(tools):`, etc.

## User Configuration

User config lives in `~/.hermes/` (not in the repo):
- `config.yaml` — settings
- `.env` — API keys
- `skills/` — active skills
- `memories/` — MEMORY.md, USER.md
- `state.db` — SQLite session database
- `sessions/` — JSON session logs

## Repository Structure Notes

- `hermes-agent/AGENTS.md` — development guide for AI coding assistants (important for prompt engineering context)
- `hermes-agent/CONTRIBUTING.md` — full contributing guide with architecture overview and tool/skill authoring patterns
- Root-level `hermes-agent v2026.4.8 源代码分析.md` — Chinese-language architecture analysis (reference material)
- `hermes-agent/website/` — Docusaurus documentation site
