#!/usr/bin/env python3
"""技能工具模块 - 技能文档的列表和查看

技能是以目录形式组织的，包含 SKILL.md 文件（主要指令）和可选的支撑文件
（references、templates、assets）。

渐进式披露架构（Anthropic 推荐）：
- Tier 1: 技能列表 - 仅返回元数据（名称、描述）
- Tier 2: 技能视图 - 加载完整 SKILL.md 内容
- Tier 3: 链接文件 - 按需加载 references、templates、scripts

SKILL.md 格式（YAML frontmatter，agentskills.io 兼容）：
    ---
    name: skill-name              # 必需，最大64字符
    description: 简要描述          # 必需，最大1024字符
    version: 1.0.0                # 可选
    platforms: [macos]            # 可选 - 限制特定平台
    prerequisites:                # 可选 - 前置要求
    required_environment_variables:  # 可选 - 所需环境变量
    ---
    # 技能标题
    完整指令内容...
"""

import json
import logging

from hermes_constants import get_hermes_home
import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# 技能目录：~/.hermes/skills/（单点真相）
HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# 平台标识符映射
_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXCLUDED_SKILL_DIRS = frozenset((".git", ".github", ".hub"))
_REMOTE_ENV_BACKENDS = frozenset({"docker", "singularity", "modal", "ssh", "daytona"})
_secret_capture_callback = None


def load_env() -> Dict[str, str]:
    """从 HERMES_HOME/.env 加载环境变量"""
    env_path = get_hermes_home() / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip("\"'")
    return env_vars


class SkillReadinessStatus(str, Enum):
    """技能就绪状态枚举"""
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


def set_secret_capture_callback(callback) -> None:
    """设置密钥捕获回调"""
    global _secret_capture_callback
    _secret_capture_callback = callback


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """检查技能是否与当前 OS 平台兼容"""
    from agent.skill_utils import skill_matches_platform as _impl
    return _impl(frontmatter)


def _normalize_prerequisite_values(value: Any) -> List[str]:
    """规范化前置条件值"""
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def _collect_prerequisite_values(
    frontmatter: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """收集 legacy 格式的前置条件值"""
    prereqs = frontmatter.get("prerequisites")
    if not prereqs or not isinstance(prereqs, dict):
        return [], []
    return (
        _normalize_prerequisite_values(prereqs.get("env_vars")),
        _normalize_prerequisite_values(prereqs.get("commands")),
    )


def _normalize_setup_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    """规范化 setup 元数据"""
    setup = frontmatter.get("setup")
    if not isinstance(setup, dict):
        return {"help": None, "collect_secrets": []}

    help_text = setup.get("help")
    normalized_help = (
        str(help_text).strip()
        if isinstance(help_text, str) and help_text.strip()
        else None
    )

    collect_secrets_raw = setup.get("collect_secrets")
    if isinstance(collect_secrets_raw, dict):
        collect_secrets_raw = [collect_secrets_raw]
    if not isinstance(collect_secrets_raw, list):
        collect_secrets_raw = []

    collect_secrets: List[Dict[str, Any]] = []
    for item in collect_secrets_raw:
        if not isinstance(item, dict):
            continue

        env_var = str(item.get("env_var") or "").strip()
        if not env_var:
            continue

        prompt = str(item.get("prompt") or f"Enter value for {env_var}").strip()
        provider_url = str(item.get("provider_url") or item.get("url") or "").strip()

        entry: Dict[str, Any] = {
            "env_var": env_var,
            "prompt": prompt,
            "secret": bool(item.get("secret", True)),
        }
        if provider_url:
            entry["provider_url"] = provider_url
        collect_secrets.append(entry)

    return {
        "help": normalized_help,
        "collect_secrets": collect_secrets,
    }


def _get_required_environment_variables(
    frontmatter: Dict[str, Any],
    legacy_env_vars: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """获取所需环境变量列表"""
    setup = _normalize_setup_metadata(frontmatter)
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, dict):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    required: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_required(entry: Dict[str, Any]) -> None:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in seen:
            return
        if not _ENV_VAR_NAME_RE.match(env_name):
            return

        normalized: Dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip(),
        }

        help_text = (
            entry.get("help")
            or entry.get("provider_url")
            or entry.get("url")
            or setup.get("help")
        )
        if isinstance(help_text, str) and help_text.strip():
            normalized["help"] = help_text.strip()

        required_for = entry.get("required_for")
        if isinstance(required_for, str) and required_for.strip():
            normalized["required_for"] = required_for.strip()

        seen.add(env_name)
        required.append(normalized)

    for item in required_raw:
        if isinstance(item, str):
            _append_required({"name": item})
            continue
        if isinstance(item, dict):
            _append_required(item)

    for item in setup["collect_secrets"]:
        _append_required(
            {
                "name": item.get("env_var"),
                "prompt": item.get("prompt"),
                "help": item.get("provider_url") or setup.get("help"),
            }
        )

    if legacy_env_vars is None:
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
    for env_var in legacy_env_vars:
        _append_required({"name": env_var})

    return required


def _capture_required_environment_variables(
    skill_name: str,
    missing_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """捕获所需环境变量"""
    if not missing_entries:
        return {
            "missing_names": [],
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    missing_names = [entry["name"] for entry in missing_entries]
    if _is_gateway_surface():
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": _gateway_setup_hint(),
        }

    if _secret_capture_callback is None:
        return {
            "missing_names": missing_names,
            "setup_skipped": False,
            "gateway_setup_hint": None,
        }

    setup_skipped = False
    remaining_names: List[str] = []

    for entry in missing_entries:
        metadata = {"skill_name": skill_name}
        if entry.get("help"):
            metadata["help"] = entry["help"]
        if entry.get("required_for"):
            metadata["required_for"] = entry["required_for"]

        try:
            callback_result = _secret_capture_callback(
                entry["name"],
                entry["prompt"],
                metadata,
            )
        except Exception:
            logger.warning(
                f"Secret capture callback failed for {entry['name']}", exc_info=True
            )
            callback_result = {
                "success": False,
                "stored_as": entry["name"],
                "validated": False,
                "skipped": True,
            }

        success = isinstance(callback_result, dict) and bool(
            callback_result.get("success")
        )
        skipped = isinstance(callback_result, dict) and bool(
            callback_result.get("skipped")
        )
        if success and not skipped:
            continue

        setup_skipped = True
        remaining_names.append(entry["name"])

    return {
        "missing_names": remaining_names,
        "setup_skipped": setup_skipped,
        "gateway_setup_hint": None,
    }


def _is_gateway_surface() -> bool:
    """检查是否在 gateway 环境中运行"""
    if os.getenv("HERMES_GATEWAY_SESSION"):
        return True
    return bool(os.getenv("HERMES_SESSION_PLATFORM"))


def _get_terminal_backend_name() -> str:
    """获取终端后端名称"""
    return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"


def _is_env_var_persisted(
    var_name: str, env_snapshot: Dict[str, str] | None = None
) -> bool:
    """检查环境变量是否已持久化"""
    if env_snapshot is None:
        env_snapshot = load_env()
    if var_name in env_snapshot:
        return bool(env_snapshot.get(var_name))
    return bool(os.getenv(var_name))


def _remaining_required_environment_names(
    required_env_vars: List[Dict[str, Any]],
    capture_result: Dict[str, Any],
    *,
    env_snapshot: Dict[str, str] | None = None,
) -> List[str]:
    """检查仍缺失的环境变量"""
    missing_names = set(capture_result["missing_names"])

    if env_snapshot is None:
        env_snapshot = load_env()
    remaining = []
    for entry in required_env_vars:
        name = entry["name"]
        if name in missing_names or not _is_env_var_persisted(name, env_snapshot):
            remaining.append(name)
    return remaining


def _gateway_setup_hint() -> str:
    """获取 gateway 设置提示"""
    try:
        from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
        return GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
    except Exception:
        return "Secure secret entry is not available. Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually."


def _build_setup_note(
    readiness_status: SkillReadinessStatus,
    missing: List[str],
    setup_help: str | None = None,
) -> str | None:
    """构建设置说明"""
    if readiness_status == SkillReadinessStatus.SETUP_NEEDED:
        missing_str = ", ".join(missing) if missing else "required prerequisites"
        note = f"Setup needed before using this skill: missing {missing_str}."
        if setup_help:
            return f"{note} {setup_help}"
        return note
    return None


def check_skills_requirements() -> bool:
    """技能始终可用 - 目录在首次使用时创建"""
    return True


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """解析 YAML frontmatter"""
    from agent.skill_utils import parse_frontmatter
    return parse_frontmatter(content)


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """从技能路径提取分类

    例如：~/.hermes/skills/mlops/axolotl/SKILL.md -> "mlops"
    """
    dirs_to_check = [SKILLS_DIR]
    try:
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    except Exception:
        pass
    for skills_dir in dirs_to_check:
        try:
            rel_path = skill_path.relative_to(skills_dir)
            parts = rel_path.parts
            if len(parts) >= 3:
                return parts[0]
        except ValueError:
            continue
    return None


def _estimate_tokens(content: str) -> int:
    """粗略 token 估算（每 token 4 字符）"""
    return len(content) // 4


def _parse_tags(tags_value) -> List[str]:
    """解析 tags 值 - 支持列表、括号字符串、逗号分隔字符串"""
    if not tags_value:
        return []

    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]

    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]

    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]


def _get_disabled_skill_names() -> Set[str]:
    """从配置加载禁用的技能名称"""
    from agent.skill_utils import get_disabled_skill_names
    return get_disabled_skill_names()


def _is_skill_disabled(name: str, platform: str = None) -> bool:
    """检查技能是否在配置中被禁用"""
    import os
    try:
        from hermes_cli.config import load_config
        config = load_config()
        skills_cfg = config.get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM")
        if resolved_platform:
            platform_disabled = skills_cfg.get("platform_disabled", {}).get(resolved_platform)
            if platform_disabled is not None:
                return name in platform_disabled
        return name in skills_cfg.get("disabled", [])
    except Exception:
        return False


def _find_all_skills(*, skip_disabled: bool = False) -> List[Dict[str, Any]]:
    """递归查找所有技能（~/.hermes/skills/ 和外部目录）"""
    from agent.skill_utils import get_external_skills_dirs

    skills = []
    seen_names: set = set()

    disabled = set() if skip_disabled else _get_disabled_skill_names()

    dirs_to_scan = []
    if SKILLS_DIR.exists():
        dirs_to_scan.append(SKILLS_DIR)
    dirs_to_scan.extend(get_external_skills_dirs())

    for scan_dir in dirs_to_scan:
        for skill_md in scan_dir.rglob("SKILL.md"):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue

            skill_dir = skill_md.parent

            try:
                content = skill_md.read_text(encoding="utf-8")[:4000]
                frontmatter, body = _parse_frontmatter(content)

                if not skill_matches_platform(frontmatter):
                    continue

                name = frontmatter.get("name", skill_dir.name)[:MAX_NAME_LENGTH]
                if name in seen_names:
                    continue
                if name in disabled:
                    continue

                description = frontmatter.get("description", "")
                if not description:
                    for line in body.strip().split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            description = line
                            break

                if len(description) > MAX_DESCRIPTION_LENGTH:
                    description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."

                category = _get_category_from_path(skill_md)

                seen_names.add(name)
                skills.append({
                    "name": name,
                    "description": description,
                    "category": category,
                })

            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
                continue
            except Exception as e:
                logger.debug(
                    "Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True
                )
                continue

    return skills


def _load_category_description(category_dir: Path) -> Optional[str]:
    """加载分类描述（从 DESCRIPTION.md）"""
    desc_file = category_dir / "DESCRIPTION.md"
    if not desc_file.exists():
        return None

    try:
        content = desc_file.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(content)

        description = frontmatter.get("description", "")
        if not description:
            for line in body.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    description = line
                    break

        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

        return description if description else None
    except (UnicodeDecodeError, PermissionError) as e:
        logger.debug("Failed to read category description %s: %s", desc_file, e)
        return None
    except Exception as e:
        logger.warning(
            "Error parsing category description %s: %s", desc_file, e, exc_info=True
        )
        return None


def skills_categories(verbose: bool = False, task_id: str = None) -> str:
    """列出可用技能分类（渐进式披露 Tier 0）"""
    try:
        all_dirs = [SKILLS_DIR] if SKILLS_DIR.exists() else []
        try:
            from agent.skill_utils import get_external_skills_dirs
            all_dirs.extend(d for d in get_external_skills_dirs() if d.exists())
        except Exception:
            pass
        if not all_dirs:
            return json.dumps(
                {
                    "success": True,
                    "categories": [],
                    "message": "No skills directory found.",
                },
                ensure_ascii=False,
            )

        category_dirs = {}
        category_counts: Dict[str, int] = {}
        for scan_dir in all_dirs:
            for skill_md in scan_dir.rglob("SKILL.md"):
                if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                    continue

                try:
                    frontmatter, _ = _parse_frontmatter(
                        skill_md.read_text(encoding="utf-8")[:4000]
                    )
                except Exception:
                    frontmatter = {}

                if not skill_matches_platform(frontmatter):
                    continue

                category = _get_category_from_path(skill_md)
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1
                    if category not in category_dirs:
                        category_dirs[category] = skill_md.parent.parent

        categories = []
        for name in sorted(category_dirs.keys()):
            category_dir = category_dirs[name]
            description = _load_category_description(category_dir)

            cat_entry = {"name": name, "skill_count": category_counts[name]}
            if description:
                cat_entry["description"] = description
            categories.append(cat_entry)

        return json.dumps(
            {
                "success": True,
                "categories": categories,
                "hint": "If a category is relevant to your task, use skills_list with that category to see available skills",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return tool_error(str(e), success=False)


def skills_list(category: str = None, task_id: str = None) -> str:
    """列出所有可用技能（渐进式披露 Tier 1 - 最小元数据）

    仅返回名称+描述以最小化 token 使用量。使用 skill_view() 加载完整内容。
    """
    try:
        if not SKILLS_DIR.exists():
            SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found. Skills directory created at ~/.hermes/skills/",
                },
                ensure_ascii=False,
            )

        all_skills = _find_all_skills()

        if not all_skills:
            return json.dumps(
                {
                    "success": True,
                    "skills": [],
                    "categories": [],
                    "message": "No skills found in skills/ directory.",
                },
                ensure_ascii=False,
            )

        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]

        all_skills.sort(key=lambda s: (s.get("category") or "", s["name"]))

        categories = sorted(
            set(s.get("category") for s in all_skills if s.get("category"))
        )

        return json.dumps(
            {
                "success": True,
                "skills": all_skills,
                "categories": categories,
                "count": len(all_skills),
                "hint": "Use skill_view(name) to see full content, tags, and linked files",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return tool_error(str(e), success=False)


def skill_view(name: str, file_path: str = None, task_id: str = None) -> str:
    """查看技能内容或技能目录中的特定文件

    渐进式披露流程：
    1. skill_view(name) -> SKILL.md 内容 + linked_files 字典
    2. skill_view(name, file_path) -> 加载特定链接文件
    """
    try:
        from agent.skill_utils import get_external_skills_dirs

        all_dirs = []
        if SKILLS_DIR.exists():
            all_dirs.append(SKILLS_DIR)
        all_dirs.extend(get_external_skills_dirs())

        if not all_dirs:
            return json.dumps(
                {
                    "success": False,
                    "error": "Skills directory does not exist yet. It will be created on first install.",
                },
                ensure_ascii=False,
            )

        skill_dir = None
        skill_md = None

        # 搜索所有目录：本地优先
        for search_dir in all_dirs:
            direct_path = search_dir / name
            if direct_path.is_dir() and (direct_path / "SKILL.md").exists():
                skill_dir = direct_path
                skill_md = direct_path / "SKILL.md"
                break
            elif direct_path.with_suffix(".md").exists():
                skill_md = direct_path.with_suffix(".md")
                break

        # 按目录名搜索
        if not skill_md:
            for search_dir in all_dirs:
                for found_skill_md in search_dir.rglob("SKILL.md"):
                    if found_skill_md.parent.name == name:
                        skill_dir = found_skill_md.parent
                        skill_md = found_skill_md
                        break
                if skill_md:
                    break

        # 遗留：扁平 .md 文件
        if not skill_md:
            for search_dir in all_dirs:
                for found_md in search_dir.rglob(f"{name}.md"):
                    if found_md.name != "SKILL.md":
                        skill_md = found_md
                        break
                if skill_md:
                    break

        if not skill_md or not skill_md.exists():
            available = [s["name"] for s in _find_all_skills()[:20]]
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' not found.",
                    "available_skills": available,
                    "hint": "Use skills_list to see all available skills",
                },
                ensure_ascii=False,
            )

        try:
            # 【文档锚点 5B】skill_view 运行时入口：真正把某个技能文件读出来提供给 agent 使用
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to read skill '{name}': {e}",
                },
                ensure_ascii=False,
            )

        # 安全检查：提示注入模式检测
        _INJECTION_PATTERNS = [
            "ignore previous instructions",
            "ignore all previous",
            "you are now",
            "disregard your",
            "forget your instructions",
            "new instructions:",
            "system prompt:",
            "<system>",
            "]]>",
        ]
        _content_lower = content.lower()
        _injection_detected = any(p in _content_lower for p in _INJECTION_PATTERNS)

        if _injection_detected:
            import logging as _logging
            _logging.getLogger(__name__).warning("Skill security warning for '%s': prompt injection detected", name)

        parsed_frontmatter: Dict[str, Any] = {}
        try:
            parsed_frontmatter, _ = _parse_frontmatter(content)
        except Exception:
            parsed_frontmatter = {}

        if not skill_matches_platform(parsed_frontmatter):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill '{name}' is not supported on this platform.",
                    "readiness_status": SkillReadinessStatus.UNSUPPORTED.value,
                },
                ensure_ascii=False,
            )

        resolved_name = parsed_frontmatter.get("name", skill_md.parent.name)
        if _is_skill_disabled(resolved_name):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Skill '{resolved_name}' is disabled. "
                        "Enable it with `hermes skills` or inspect the files directly on disk."
                    ),
                },
                ensure_ascii=False,
            )

        # 请求特定文件路径时读取该文件
        if file_path and skill_dir:
            normalized_path = Path(file_path)
            if ".." in normalized_path.parts:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Path traversal ('..') is not allowed.",
                        "hint": "Use a relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )

            target_file = skill_dir / file_path

            try:
                resolved = target_file.resolve()
                skill_dir_resolved = skill_dir.resolve()
                if not resolved.is_relative_to(skill_dir_resolved):
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Path escapes skill directory boundary.",
                            "hint": "Use a relative path within the skill directory",
                        },
                        ensure_ascii=False,
                    )
            except (OSError, ValueError):
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Invalid file path: '{file_path}'",
                        "hint": "Use a valid relative path within the skill directory",
                    },
                    ensure_ascii=False,
                )
            if not target_file.exists():
                available_files = {
                    "references": [],
                    "templates": [],
                    "assets": [],
                    "scripts": [],
                    "other": [],
                }

                for f in skill_dir.rglob("*"):
                    if f.is_file() and f.name != "SKILL.md":
                        rel = str(f.relative_to(skill_dir))
                        if rel.startswith("references/"):
                            available_files["references"].append(rel)
                        elif rel.startswith("templates/"):
                            available_files["templates"].append(rel)
                        elif rel.startswith("assets/"):
                            available_files["assets"].append(rel)
                        elif rel.startswith("scripts/"):
                            available_files["scripts"].append(rel)
                        elif f.suffix in [
                            ".md", ".py", ".yaml", ".yml", ".json", ".tex", ".sh",
                        ]:
                            available_files["other"].append(rel)

                available_files = {k: v for k, v in available_files.items() if v}

                return json.dumps(
                    {
                        "success": False,
                        "error": f"File '{file_path}' not found in skill '{name}'.",
                        "available_files": available_files,
                        "hint": "Use one of the available file paths listed above",
                    },
                    ensure_ascii=False,
                )

            try:
                content = target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return json.dumps(
                    {
                        "success": True,
                        "name": name,
                        "file": file_path,
                        "content": f"[Binary file: {target_file.name}, size: {target_file.stat().st_size} bytes]",
                        "is_binary": True,
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "success": True,
                    "name": name,
                    "file": file_path,
                    "content": content,
                    "file_type": target_file.suffix,
                },
                ensure_ascii=False,
            )

        frontmatter = parsed_frontmatter

        # 获取链接文件列表
        reference_files = []
        template_files = []
        asset_files = []
        script_files = []

        if skill_dir:
            references_dir = skill_dir / "references"
            if references_dir.exists():
                reference_files = [
                    str(f.relative_to(skill_dir)) for f in references_dir.glob("*.md")
                ]

            templates_dir = skill_dir / "templates"
            if templates_dir.exists():
                for ext in ["*.md", "*.py", "*.yaml", "*.yml", "*.json", "*.tex", "*.sh"]:
                    template_files.extend(
                        [
                            str(f.relative_to(skill_dir))
                            for f in templates_dir.rglob(ext)
                        ]
                    )

            assets_dir = skill_dir / "assets"
            if assets_dir.exists():
                for f in assets_dir.rglob("*"):
                    if f.is_file():
                        asset_files.append(str(f.relative_to(skill_dir)))

            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists():
                for ext in ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"]:
                    script_files.extend(
                        [str(f.relative_to(skill_dir)) for f in scripts_dir.glob(ext)]
                    )

        # 解析 tags 和 related_skills
        hermes_meta = {}
        metadata = frontmatter.get("metadata")
        if isinstance(metadata, dict):
            hermes_meta = metadata.get("hermes", {}) or {}

        tags = _parse_tags(hermes_meta.get("tags") or frontmatter.get("tags", ""))
        related_skills = _parse_tags(
            hermes_meta.get("related_skills") or frontmatter.get("related_skills", "")
        )

        linked_files = {}
        if reference_files:
            linked_files["references"] = reference_files
        if template_files:
            linked_files["templates"] = template_files
        if asset_files:
            linked_files["assets"] = asset_files
        if script_files:
            linked_files["scripts"] = script_files

        try:
            rel_path = str(skill_md.relative_to(SKILLS_DIR))
        except ValueError:
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get(
            "name", skill_md.stem if not skill_dir else skill_dir.name
        )
        legacy_env_vars, _ = _collect_prerequisite_values(frontmatter)
        required_env_vars = _get_required_environment_variables(
            frontmatter, legacy_env_vars
        )
        backend = _get_terminal_backend_name()
        env_snapshot = load_env()
        missing_required_env_vars = [
            e
            for e in required_env_vars
            if not _is_env_var_persisted(e["name"], env_snapshot)
        ]
        capture_result = _capture_required_environment_variables(
            skill_name,
            missing_required_env_vars,
        )
        if missing_required_env_vars:
            env_snapshot = load_env()
        remaining_missing_required_envs = _remaining_required_environment_names(
            required_env_vars,
            capture_result,
            env_snapshot=env_snapshot,
        )
        setup_needed = bool(remaining_missing_required_envs)

        # 注册可用的技能环境变量以便传递到沙箱执行环境
        available_env_names = [
            e["name"]
            for e in required_env_vars
            if e["name"] not in remaining_missing_required_envs
        ]
        if available_env_names:
            try:
                from tools.env_passthrough import register_env_passthrough
                register_env_passthrough(available_env_names)
            except Exception:
                logger.debug(
                    "Could not register env passthrough for skill %s",
                    skill_name,
                    exc_info=True,
                )

        # 注册凭证文件以挂载到远程沙箱
        required_cred_files_raw = frontmatter.get("required_credential_files", [])
        if not isinstance(required_cred_files_raw, list):
            required_cred_files_raw = []
        missing_cred_files: list = []
        if required_cred_files_raw:
            try:
                from tools.credential_files import register_credential_files
                missing_cred_files = register_credential_files(required_cred_files_raw)
                if missing_cred_files:
                    setup_needed = True
            except Exception:
                logger.debug(
                    "Could not register credential files for skill %s",
                    skill_name,
                    exc_info=True,
                )

        result = {
            "success": True,
            "name": skill_name,
            "description": frontmatter.get("description", ""),
            "tags": tags,
            "related_skills": related_skills,
            "content": content,
            "path": rel_path,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'"
            if linked_files
            else None,
            "required_environment_variables": required_env_vars,
            "required_commands": [],
            "missing_required_environment_variables": remaining_missing_required_envs,
            "missing_credential_files": missing_cred_files,
            "missing_required_commands": [],
            "setup_needed": setup_needed,
            "setup_skipped": capture_result["setup_skipped"],
            "readiness_status": SkillReadinessStatus.SETUP_NEEDED.value
            if setup_needed
            else SkillReadinessStatus.AVAILABLE.value,
        }

        setup_help = next((e["help"] for e in required_env_vars if e.get("help")), None)
        if setup_help:
            result["setup_help"] = setup_help

        if capture_result["gateway_setup_hint"]:
            result["gateway_setup_hint"] = capture_result["gateway_setup_hint"]

        if setup_needed:
            missing_items = [
                f"env ${env_name}" for env_name in remaining_missing_required_envs
            ] + [
                f"file {path}" for path in missing_cred_files
            ]
            setup_note = _build_setup_note(
                SkillReadinessStatus.SETUP_NEEDED,
                missing_items,
                setup_help,
            )
            if backend in _REMOTE_ENV_BACKENDS and setup_note:
                setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
            if setup_note:
                result["setup_note"] = setup_note

        if frontmatter.get("compatibility"):
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return tool_error(str(e), success=False)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

# 【文档锚点 5B】技能运行时入口：skills_list / skill_view 暴露“发现技能 / 加载技能内容”的工具面

SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills)",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(
        category=args.get("category"), task_id=kw.get("task_id")
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=lambda args, **kw: skill_view(
        args.get("name", ""), file_path=args.get("file_path"), task_id=kw.get("task_id")
    ),
    check_fn=check_skills_requirements,
    emoji="📚",
)
