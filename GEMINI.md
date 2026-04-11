# Gemini Instructional Context - Hello Hermes ☤

This workspace is dedicated to the development, analysis, and exploration of the **Hermes Agent** (specifically version `v0.8.0 / v2026.4.8`). It contains the core `hermes-agent` source code and high-level architectural analysis.

## Project Overview

- **Purpose:** A self-improving AI agent system featuring built-in learning loops (memories/skills), multi-platform messaging (Telegram, Discord, etc.), and a powerful terminal interface.
- **Main Technologies:**
  - **Python 3.11+**: Primary language (AIAgent, CLI, Gateway).
  - **uv**: Fast Python package manager for environment and dependency management.
  - **SQLite**: Underlying database for session and state persistence (`FTS5` enabled).
  - **Core Libs**: `pydantic` (schemas), `tenacity` (retries), `jinja2` (prompts), `fire` (CLI), `prompt_toolkit` (REPL).
  - **Optional Integrations**: `modal`, `daytona`, `mcp`, `faster-whisper`, `elevenlabs`.
  - **Tools**: `exa-py`, `firecrawl-py`, `parallel-web`.
  - **Node.js/Docusaurus**: Used for the documentation website.

## Architecture Summary

Hermes is a layered agent system (6 layers):

1.  **Entry Layer**: Entry points in `pyproject.toml`:
    - `hermes` → `hermes_cli.main:main` (Unified CLI)
    - `hermes-agent` → `run_agent:main` (Agent kernel)
2.  **Control Layer**: `hermes_cli/main.py` handles profiles, environments, and command distribution.
3.  **Shell Layer**: `cli.py` (Interactive REPL) and `gateway/run.py` (Messaging adapters).
4.  **Core Orchestrator**: `run_agent.py:AIAgent` manages the LLM loop, tool execution, and memory.
5.  **Capability Layer**:
    - `agent/`: Prompting, memory, context compression.
    - `tools/`: Self-registering tools (via `registry.register()`).
    - `toolsets.py`: Logical grouping of tools.
6.  **Persistence Layer**: `hermes_state.py` (SQLite `SessionDB`) and JSON session log files.

### Call Chain (REPL)
`hermes` -> `hermes_cli.main:main()` -> `cmd_chat()` -> `cli.main()` -> `HermesCLI` -> `AIAgent`

## Development & Operations

### Building and Running
Always operate from the `hermes-agent/` directory for code-related tasks.

```bash
cd hermes-agent
# Setup environment
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"

# Run doctor for health checks
hermes doctor

# Run agent CLI locally
python -c "from cli import main; main()"
```

### Testing
- **Test Runner**: `pytest` (configured in `pyproject.toml`).
- **Execution**: `python -m pytest tests/` (skips integration tests by default).
- **Conventions**: New features/fixes MUST include corresponding tests in `tests/`.

### Coding Conventions
- **Style**: 4-space indentation, `snake_case` (functions/modules), `PascalCase` (classes).
- **Security**: Always use `shlex.quote()` for shell interpolation. Use `pathlib.Path` for cross-platform file handling.
- **Commits**: Use Conventional Commits (`feat(cli): ...`, `fix(agent): ...`).
- **Documentation**: Match existing patterns in the module being edited.

## Interaction Guidelines

### Commit Identity & Co-Authorship (MANDATORY)
When committing as Gemini, use the following identity:
```bash
git commit -m "<type>(<scope>): <message>" \
  --author="Gemini <noreply@google.com>" \
  -m "Co-authored-by: Gemini <noreply@google.com>"
```

### Sub-Agent Usage
- **codebase_investigator**: For deep architecture mapping or bug root-cause analysis.
- **generalist**: For high-volume batch refactoring or speculative research.

### Diagram Generation (Quality Rules)
When using diagram skills, strictly follow `AGENTS.md` quality rules:
- **No Text Overflow**: Box width must be at least 2x text width; min 30px horizontal padding.
- **Complete Arrows**: Path must extend at least 15px past marker refX.
- **Stay in ViewBox**: Loopback paths must not be truncated at `y=0`.

## Key Reference Files
- `README.md`: High-level project intro.
- `AGENTS.md`: **CRITICAL** Detailed repository guidelines, commit rules, and diagram standards.
- `CLAUDE.md`: Technical patterns and call chains (also relevant for Gemini).
- `hermes-agent v2026.4.8 源代码分析.md`: Deep-dive architectural mapping (Chinese).
- `hermes-agent/CONTRIBUTING.md`: Full guide for Tool/Skill authoring.
