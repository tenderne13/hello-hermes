# Gemini Instructional Context - Hello Hermes ☤

This workspace is dedicated to the development, analysis, and exploration of the **Hermes Agent** (specifically version `v0.8.0 / v2026.4.8`). It contains the core `hermes-agent` source code and high-level architectural analysis.

## Project Overview

- **Purpose:** A self-improving AI agent system featuring built-in learning loops (memories/skills), multi-platform messaging (Telegram, Discord, etc.), and a powerful terminal interface.
- **Main Technologies:**
  - **Python 3.11+**: Primary language for the agent core, CLI, and gateway.
  - **uv**: Fast Python package manager used for environment and dependency management.
  - **Node.js**: Used for the documentation website (Docusaurus).
  - **SQLite**: Underlying database for session and state persistence.
  - **Tools & Frameworks**: `openai`, `anthropic`, `prompt_toolkit`, `rich`, `pytest`, `mcp`.

## Architecture Summary

The system is organized into distinct layers:
1.  **Entry Layer**: `hermes` CLI launcher and `pyproject.toml` scripts.
2.  **Control Layer**: `hermes_cli/main.py` handles profiles, envs, and command distribution.
3.  **Shell Layer**: `cli.py` (Interactive REPL) and `gateway/run.py` (Messaging adapters).
4.  **Core Orchestrator**: `run_agent.py:AIAgent` manages the LLM loop, tool execution, and memory.
5.  **Capability Layer**: `agent/` (prompting, memory, compression), `tools/` (tool registry and logic).
6.  **Persistence Layer**: `hermes_state.py` (SQLite) and session log files.

## Development & Operations

### Building and Running
Always operate from the `hermes-agent/` directory for code-related tasks.

```bash
cd hermes-agent
# Setup environment
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"

# Run agent CLI
hermes

# Run doctor for health checks
hermes doctor
```

### Testing
- **Test Runner**: `pytest`
- **Execution**: `python -m pytest tests/ -q --ignore=tests/integration --ignore=tests/e2e --tb=short -n auto`
- **Conventions**: New features or bug fixes MUST include corresponding tests in `tests/`. Integration tests require external API keys and are skipped by default.

### Coding Conventions
- **Style**: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- **Documentation**: Concise docstrings for non-obvious logic. Match existing patterns in the module being edited.
- **Commits**: Use Conventional Commits (e.g., `feat(cli): ...`, `fix(agent): ...`).

## Key Reference Files
- `README.md`: High-level project intro and quickstart.
- `AGENTS.md`: Detailed repository guidelines, build commands, and diagram generation rules.
- `hermes-agent v2026.4.8 源代码分析.md`: Deep-dive architectural mapping and line-by-line code pointers.
- `hermes-agent/pyproject.toml`: Dependency definitions and entry points.

## Interaction Guidelines
- **Sub-Agent Usage**: For large-scale refactors or deep investigations, delegate to `codebase_investigator` or `generalist`.
- **Diagrams**: Follow strict quality rules in `AGENTS.md` when using diagram skills (avoid truncated arrows, ensure text fits boxes).
- **Tool Logic**: When adding tools, register them via `tools/registry.py` for automatic discovery.
