"""工具注册中心 - Hermes Agent 工具子系统核心

工具子系统采用自注册模式：每个工具文件在模块级别调用 registry.register() 注册
其 schema、handler、工具集归属和可用性检查。model_tools.py 查询注册中心而非维护
独立的并行数据结构。

导入链（避免循环导入）：
    tools/registry.py  (不导入 model_tools 或工具文件)
           ↑
    tools/*.py  (在模块级别导入 tools.registry)
           ↑
    model_tools.py  (导入 tools.registry + 所有工具模块)
           ↑
    run_agent.py, cli.py, batch_runner.py 等
"""

import json
import logging
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ToolEntry:
    """单个已注册工具的元数据"""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars


class ToolRegistry:
    """工具注册中心单例 - 收集工具 schema + handler"""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._toolset_checks: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # 注册相关
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
    ):
        """注册工具 - 各工具文件在模块导入时调用"""
        # 【文档锚点 3D】工具模块在 import 时通过 registry.register() 完成自注册
        existing = self._tools.get(name)
        if existing and existing.toolset != toolset:
            logger.warning(
                "Tool name collision: '%s' (toolset '%s') is being "
                "overwritten by toolset '%s'",
                name, existing.toolset, toolset,
            )
        self._tools[name] = ToolEntry(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env or [],
            is_async=is_async,
            description=description or schema.get("description", ""),
            emoji=emoji,
            max_result_size_chars=max_result_size_chars,
        )
        if check_fn and toolset not in self._toolset_checks:
            self._toolset_checks[toolset] = check_fn

    def deregister(self, name: str) -> None:
        """注销工具 - MCP 动态工具发现时用于清除并重新注册"""
        entry = self._tools.pop(name, None)
        if entry is None:
            return
        if entry.toolset in self._toolset_checks and not any(
            e.toolset == entry.toolset for e in self._tools.values()
        ):
            self._toolset_checks.pop(entry.toolset, None)
        logger.debug("Deregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Schema 查询
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """返回 OpenAI 格式的工具 schema - 仅返回 check_fn() 通过的工具"""
        # 【文档锚点 3D】从注册中心取 schema 时，顺带执行可用性检查
        result = []
        check_results: Dict[Callable, bool] = {}
        for name in sorted(tool_names):
            entry = self._tools.get(name)
            if not entry:
                continue
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    try:
                        check_results[entry.check_fn] = bool(entry.check_fn())
                    except Exception:
                        check_results[entry.check_fn] = False
                        if not quiet:
                            logger.debug("Tool %s check raised; skipping", name)
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            schema_with_name = {**entry.schema, "name": entry.name}
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # 分发执行
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """按名称执行工具 handler - 异步 handler 自动桥接"""
        # 【文档锚点 3D】运行时分发：把函数名映射到真正的 Python handler
        entry = self._tools.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            if entry.is_async:
                from model_tools import _run_async
                return _run_async(entry.handler(args, **kwargs))
            return entry.handler(args, **kwargs)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            return json.dumps({"error": f"Tool execution failed: {type(e).__name__}: {e}"})

    # ------------------------------------------------------------------
    # 查询辅助方法
    # ------------------------------------------------------------------

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """返回工具最大结果大小"""
        entry = self._tools.get(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> List[str]:
        """返回所有已注册工具名称（排序）"""
        return sorted(self._tools.keys())

    def get_schema(self, name: str) -> Optional[dict]:
        """返回工具原始 schema dict - 不经过 check_fn 过滤"""
        entry = self._tools.get(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """返回工具所属的工具集"""
        entry = self._tools.get(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """返回工具的 emoji 图标"""
        entry = self._tools.get(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """返回 {tool_name: toolset_name} 映射"""
        return {name: e.toolset for name, e in self._tools.items()}

    def is_toolset_available(self, toolset: str) -> bool:
        """检查工具集是否满足要求条件"""
        check = self._toolset_checks.get(toolset)
        if not check:
            return True
        try:
            return bool(check())
        except Exception:
            logger.debug("Toolset %s check raised; marking unavailable", toolset)
            return False

    def check_toolset_requirements(self) -> Dict[str, bool]:
        """返回所有工具集的可用性状态"""
        toolsets = set(e.toolset for e in self._tools.values())
        return {ts: self.is_toolset_available(ts) for ts in sorted(toolsets)}

    def get_available_toolsets(self) -> Dict[str, dict]:
        """返回工具集元数据 - 用于 UI 显示"""
        toolsets: Dict[str, dict] = {}
        for entry in self._tools.values():
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": self.is_toolset_available(ts),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """构建 TOOLSET_REQUIREMENTS 兼容字典"""
        result: Dict[str, dict] = {}
        for entry in self._tools.values():
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": self._toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    def check_tool_availability(self, quiet: bool = False):
        """返回 (可用工具集, 不可用信息)"""
        available = []
        unavailable = []
        seen = set()
        for entry in self._tools.values():
            ts = entry.toolset
            if ts in seen:
                continue
            seen.add(ts)
            if self.is_toolset_available(ts):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": entry.requires_env,
                    "tools": [e.name for e in self._tools.values() if e.toolset == ts],
                })
        return available, unavailable


# 模块级单例
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# 工具响应序列化辅助函数
# 所有工具 handler 必须返回 JSON 字符串
# ---------------------------------------------------------------------------

def tool_error(message, **extra) -> str:
    """返回工具错误 JSON 字符串"""
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """返回工具结果 JSON 字符串"""
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)
