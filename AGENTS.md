# Repository Guidelines

## Project Structure & Module Organization

This repository is a workspace around the upstream Hermes Agent source. Do most development inside `hermes-agent/`. Core Python packages live in `hermes-agent/agent`, `hermes-agent/tools`, `hermes-agent/hermes_cli`, `hermes-agent/gateway`, `hermes-agent/cron`, and `hermes-agent/acp_adapter`. Tests are under `hermes-agent/tests`, grouped by subsystem such as `tests/cli`, `tests/gateway`, and `tests/tools`. Documentation code lives in `hermes-agent/website`, while repo-level notes like `README.md`, `images/`, and `hermes-agent源代码分析.md` are reference material, not the main product.

## Build, Test, and Development Commands

Run commands from `hermes-agent/` unless you are editing root-level notes.

```bash
cd hermes-agent
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
npm install
python -m pytest tests/ -q --ignore=tests/integration --ignore=tests/e2e --tb=short -n auto
```

Use `hermes doctor` for a local sanity check after installing. For docs work, run `cd hermes-agent/website && npm install && npm run start`; use `npm run build`, `npm run typecheck`, and `npm run lint:diagrams` before merging website changes.

## Coding Style & Naming Conventions

Python is the primary language. Follow existing style: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and concise docstrings where behavior is not obvious. Match nearby patterns instead of introducing a new formatter style; no repo-wide Black or Ruff config is checked in here. Keep modules focused by subsystem and place new tests beside the affected area. In `website/`, keep TypeScript and Docusaurus files consistent with the existing naming and directory structure.

## Testing Guidelines

Pytest is the test runner, with `pytest-xdist` enabled by default and `integration` tests excluded unless explicitly targeted. Name tests `test_*.py`, mirroring the package under test, for example `tests/cli/test_cli_init.py`. Add or update tests for every behavior change, especially around CLI flow, tool dispatch, config migration, and cross-platform file or process handling.

## Commit & Pull Request Guidelines

The active history in `hermes-agent/` uses Conventional Commits such as `fix(process): ...` and `feat(cli): ...`; continue that pattern. PRs should follow `.github/PULL_REQUEST_TEMPLATE.md`: link the issue, summarize the change, provide clear reproduction or verification steps, and note documentation or config updates. Include screenshots or logs when UI, CLI output, or docs rendering changes.

## Diagram Generation (fireworks-tech-graph Skill)

When generating diagrams using the `fireworks-tech-graph` skill, **strictly observe these quality rules**:

### Line & Arrow Quality (CRITICAL - Negative Examples Observed)

- **路径必须连接到形状边界，不能停在半空**: When a path goes from a diamond decision to a box, the horizontal line must reach the box's left or right edge. A common defect is: `d="M 594 836 L 448 836"` where 448 is the box's left edge but the path stops short at some intermediate point. The path should be: `d="M 588 836 L 144 836 L 144 866"` where the line actually reaches the box boundary at x=144.
  - Wrong: `d="M 594 836 L 448 836"` (stops short of box at x=124)
  - Correct: `d="M 588 836 L 124 836 L 124 857"` (reaches the box edge)
- **垂直路径向下延伸时必须到达下一个形状**: When routing from a diamond downward, the vertical path must extend to the next shape's boundary, not stop midway. Example: from `y=1372` diamond to `y=1516` box should have `d="M 700 1484 L 700 1516"` where 1516 is the box's top Y.
- **向上走的loopback路径不能在viewBox边界截断**: When a path goes upward (retry/fallback loop), ensure `d` starts below viewBox top or use a U-turn that stays visible. A common defect: path going to `y=-10` before turning, which gets clipped at `y=0`.
  - Wrong: `d="M 924 1878 L 840 1878 L 840 1618 L 970 1618"` where 1618 is inside the viewBox but the upward segment from 1878 to 1618 passes outside visible area
  - Correct: Route the loopback to stay within viewBox, e.g., turn at y=1740 instead of going to 1618, or extend viewBox height
- **所有箭头必须完整**: All arrowheads must be fully rendered. The path must extend **at least 15px past** the marker reference point (refX). Truncated arrows are unacceptable.
- **正交路由优先**: Prefer orthogonal paths (horizontal + vertical segments) over diagonal lines. Use `L` commands for right-angle routing.

### Text & Box Sizing (CRITICAL - Negative Examples Observed)

- **文字绝对不能超出框**: Every `<text>` element must fit entirely within its parent `<rect>`. This is non-negotiable.
  - Box width must be **at least 2x the text width** (measured in px at the given font-size), not 1.5x
  - Add horizontal padding of **at least 30px** inside the box (not 20px)
  - If a label is 100px wide in a 12px font, the box must be at least 230px wide
- **手动设置 text-anchor**: Always set `text-anchor="middle"` for centered labels and position `x` at the box center.
- **文字换行**: For multi-word labels, use `<tspan>` for line breaks, or split into separate `<text>` elements. Never let text overflow the box boundary.

### Coordinate & Spacing Rules

- **节点间距 ≥ 80px**: Minimum 80px between node edges (increased from 60px) for proper arrow routing.
- **对齐网格**: Snap node centers to 120px horizontal intervals and 80px vertical intervals.
- **Group padding**: Add `transform="translate(x, y)"` to `<g>` groups, never hardcode absolute coordinates on child elements.

### Output Format

- **只生成 SVG**: Always output the `.svg` file only. Do NOT run `rsvg-convert` or generate PNG files.
- **SVG 直接可用**: The SVG must be valid, openable in browsers, and have correct `viewBox`, `xmlns`, and embedded styles.
- **字体嵌入**: Use `<style>` with inline font-family (no `@import`) for cross-browser compatibility.

### Example of Correct vs Incorrect Sizing

```xml
<!-- WRONG: text overflows box — THIS CAUSES VISUAL GLITCHES -->
<rect x="0" y="0" width="80" height="30"/>
<text x="40" y="20" font-size="12">A very long label</text>

<!-- CORRECT: box is 2x wider than text, proper padding -->
<rect x="0" y="0" width="240" height="50" rx="6"/>
<text x="120" y="29" text-anchor="middle" font-size="12">A very long label</text>
```

### Quality Checklist Before Saving

- [ ] All arrows have complete arrowheads with sufficient path extension past refX
- [ ] No text element overflows its parent box
- [ ] All connecting lines have adequate length (not truncated due to space constraints)
- [ ] Loopback paths stay within viewBox bounds
- [ ] Box padding is at least 30px horizontal

### Output

- Default output: `./images/[derived-name].svg` in the current working directory.
- Custom path: user specifies with `--output /path/`.
