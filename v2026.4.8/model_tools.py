#!/usr/bin/env python3
"""模型工具模块 - 工具注册中心的编排层

每个工具文件通过 tools.registry.register() 自注册其 schema、handler 和元数据。
本模块触发发现（通过导入所有工具模块），然后提供 run_agent.py、cli.py、
batch_runner.py 和 RL 环境使用的公共 API。
"""

import json
import asyncio
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import registry
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)


# =============================================================================
# 异步桥接 - 同步/异步工具 handler 的统一调用入口
# =============================================================================

_tool_loop = None          # CLI 主线程的持久化事件循环
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()  # 每个 worker 线程的持久化循环


def _get_tool_loop():
    """返回主线程的持久化事件循环 - 避免"Event loop is closed"错误

    使用持久化循环（而非每次创建新循环的 asyncio.run()）可防止缓存的
    httpx/AsyncOpenAI 客户端在垃圾回收时尝试关闭已死循环。
    """
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """返回当前 worker 线程的持久化事件循环

    每个 worker 线程（如 delegate_task 的 ThreadPoolExecutor 线程）获得
    自己存储在 thread-local 的持久化循环，避免与主线程共享 loop 竞争。
    """
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """从同步上下文运行异步协程

    内部已有事件循环时（gateway、RL 环境）在新线程中运行；
    CLI 路径使用持久化主循环；worker 线程使用 per-thread 持久化循环。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 内部已有异步上下文 - 在新线程中运行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)

    # Worker 线程使用每线程持久化循环
    if threading.current_thread() is not threading.main_thread():
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)


# =============================================================================
# 工具发现 - 导入各模块触发其 registry.register() 调用
# =============================================================================

def _discover_tools():
    """导入所有工具模块以触发其 registry.register() 调用

    使用函数包装使可选工具的导入错误（如 fal_client 未安装）不会阻止其他工具加载。
    """
    # 【文档锚点 3D】工具注册阶段：通过导入 tools.* 触发模块级自注册
    _modules = [
        "tools.web_tools",
        "tools.terminal_tool",
        "tools.file_tools",
        "tools.vision_tools",
        "tools.mixture_of_agents_tool",
        "tools.image_generation_tool",
        "tools.skills_tool",
        "tools.skill_manager_tool",
        "tools.browser_tool",
        "tools.cronjob_tools",
        "tools.rl_training_tool",
        "tools.tts_tool",
        "tools.todo_tool",
        "tools.memory_tool",
        "tools.session_search_tool",
        "tools.clarify_tool",
        "tools.code_execution_tool",
        "tools.delegate_tool",
        "tools.process_registry",
        "tools.send_message_tool",
        "tools.homeassistant_tool",
    ]
    import importlib
    for mod_name in _modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)


_discover_tools()

# MCP 工具发现
try:
    from tools.mcp_tool import discover_mcp_tools
    discover_mcp_tools()
except Exception as e:
    logger.debug("MCP tool discovery failed: %s", e)

# 插件工具发现
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


# =============================================================================
# 向后兼容常量 - 发现后构建
# =============================================================================

TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# 最近一次 get_tool_definitions() 调用的解析工具名
_last_resolved_tool_names: List[str] = []


# =============================================================================
# 遗留工具集名称映射
# =============================================================================

_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "moa_tools": ["mixture_of_agents"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "cronjob_tools": ["cronjob"],
    "rl_tools": [
        "rl_list_environments", "rl_select_environment",
        "rl_get_current_config", "rl_edit_config",
        "rl_start_training", "rl_check_status",
        "rl_stop_training", "rl_get_results",
        "rl_list_runs", "rl_test_inference"
    ],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions — 主要 schema 提供函数
# =============================================================================

def get_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    quiet_mode: bool = False,
) -> List[Dict[str, Any]]:
    """根据 toolset 过滤获取模型 API 调用的工具定义

    所有工具必须属于某个 toolset 才能访问。
    """
    # 【文档锚点 3D】工具可见面阶段：按 toolset 与 check_fn 过滤出本会话真实可见工具
    tools_to_include: set = set()

    if enabled_toolsets is not None:
        for toolset_name in enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            else:
                if not quiet_mode:
                    print(f"⚠️  Unknown toolset: {toolset_name}")

    elif disabled_toolsets:
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            else:
                if not quiet_mode:
                    print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # 向注册中心请求 schema（仅返回 check_fn 通过的工具）
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)

    # 实际通过 check_fn 过滤的工具名集合
    available_tool_names = {t["function"]["name"] for t in filtered_tools}

    # 重建 execute_code schema 仅列出实际可用的沙箱工具
    if "execute_code" in available_tool_names:
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
        dynamic_schema = build_execute_code_schema(sandbox_enabled)
        for i, td in enumerate(filtered_tools):
            if td.get("function", {}).get("name") == "execute_code":
                filtered_tools[i] = {"type": "function", "function": dynamic_schema}
                break

    # 当 web_search/web_extract 不可用时从 browser_navigate 描述中移除交叉引用
    if "browser_navigate" in available_tool_names:
        web_tools_available = {"web_search", "web_extract"} & available_tool_names
        if not web_tools_available:
            for i, td in enumerate(filtered_tools):
                if td.get("function", {}).get("name") == "browser_navigate":
                    desc = td["function"].get("description", "")
                    desc = desc.replace(
                        " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
                        "",
                    )
                    filtered_tools[i] = {
                        "type": "function",
                        "function": {**td["function"], "description": desc},
                    }
                    break

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    return filtered_tools


# =============================================================================
# handle_function_call — 主要分发器
# =============================================================================

# 需要 agent loop 拦截处理的工具（需要 agent 级状态）
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# =========================================================================
# 工具参数类型强制转换
# =========================================================================

def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """将工具调用参数强制转换为 JSON Schema 声明的类型

    LLM 常将数字返回为字符串（"42" 而非 42），将布尔值返回为字符串（"true" 而非 true）。
    本函数比较参数值与工具 registered JSON Schema，尝试安全转换。
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    for key, value in args.items():
        if not isinstance(value, str):
            continue
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")
        if not expected:
            continue
        coerced = _coerce_value(value, expected)
        if coerced is not value:
            args[key] = coerced

    return args


def _coerce_value(value: str, expected_type):
    """尝试将字符串值强制转换为期望类型"""
    if isinstance(expected_type, list):
        # 联合类型 - 依次尝试
        for t in expected_type:
            result = _coerce_value(value, t)
            if result is not value:
                return result
        return value

    if expected_type in ("integer", "number"):
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """尝试将字符串解析为数字"""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    if f != f or f == float("inf") or f == float("-inf"):
        return f
    if f == int(f):
        return int(f)
    if integer_only:
        return value
    return f


def _coerce_boolean(value: str):
    """尝试将字符串解析为布尔值"""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
) -> str:
    """主要函数调用分发器 - 将调用路由到工具注册中心"""
    # 【文档锚点 3D】通用工具最终都会回落到这里，再由 registry.dispatch() 找到真实 handler
    # 字符串参数类型强制转换
    function_args = coerce_tool_args(function_name, function_args)

    # 非读/搜索工具运行时重置连续计数器
    if function_name not in _READ_SEARCH_TOOLS:
        try:
            from tools.file_tools import notify_other_tool_call
            notify_other_tool_call(task_id or "default")
        except Exception:
            pass

    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return json.dumps({"error": f"{function_name} must be handled by the agent loop"})

        try:
            from hermes_cli.plugins import invoke_hook
            invoke_hook(
                "pre_tool_call",
                tool_name=function_name,
                args=function_args,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
            )
        except Exception:
            pass

        if function_name == "execute_code":
            sandbox_enabled = enabled_tools if enabled_tools is not None else _last_resolved_tool_names
            result = registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                enabled_tools=sandbox_enabled,
            )
        else:
            result = registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                user_task=user_task,
            )

        try:
            from hermes_cli.plugins import invoke_hook
            invoke_hook(
                "post_tool_call",
                tool_name=function_name,
                args=function_args,
                result=result,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
            )
        except Exception:
            pass

        return result

    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)


# =============================================================================
# 向后兼容包装函数
# =============================================================================

def get_all_tool_names() -> List[str]:
    """返回所有已注册工具名称"""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """返回工具所属的工具集"""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """返回工具集可用性信息"""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """返回所有已注册工具集的可用性"""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """返回 (可用工具集, 不可用信息)"""
    return registry.check_tool_availability(quiet=quiet)
