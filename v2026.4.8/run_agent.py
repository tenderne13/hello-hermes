#!/usr/bin/env python3
"""
带工具调用的 AI Agent 编排内核

本模块提供了一个独立的 Agent 内核，支持执行具备工具调用能力的 AI 模型。
它负责管理主循环、工具执行以及响应管理。

核心特性：
- 自动化的工具调用循环，直到任务完成
- 可配置的模型参数
- 错误处理与恢复机制
- 会话历史管理
- 支持多种模型供应商 (Provider)
"""

import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home

# 从 ~/.hermes/.env 加载环境变量（优先），再从项目根目录作为开发回退。
# 用户管理的 env 文件应优先于旧的 shell 导出（重启时生效）。
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")


from model_tools import (
    get_tool_definitions,
    get_toolset_for_tool,
    handle_function_call,
    check_toolset_requirements,
)
from tools.terminal_tool import cleanup_vm, get_active_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


from hermes_constants import OPENROUTER_BASE_URL

from agent.memory_manager import build_memory_context_block
from agent.retry_utils import jittered_backoff
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
    MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
    build_nous_subscription_prompt,
)
from agent.model_metadata import (
    fetch_model_metadata,
    estimate_tokens_rough, estimate_messages_tokens_rough, estimate_request_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error,
    save_context_length, is_local_endpoint,
    query_ollama_num_ctx,
)
from agent.context_compressor import ContextCompressor
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import build_skills_system_prompt, build_context_files_prompt, load_soul_md, TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS, DEVELOPER_ROLE_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from agent.display import (
    KawaiiSpinner, build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)
from agent.trajectory import (
    convert_scratchpad_to_think, has_incomplete_scratchpad,
    save_trajectory as _save_trajectory_to_file,
)
from utils import atomic_json_write, env_var_enabled



class _SafeWriter:
    """透明的标准 I/O 包装器，用于捕获断开管道引发的 OSError/ValueError。

    当 hermes-agent 作为后台服务运行（如 systemd 或 Docker）时，stdout/stderr 管道可能变得不可用。
    此时任何 print() 调用都可能抛出 I/O 错误导致内核崩溃。
    
    此外，当子代理在线程池中运行时，共享的 stdout 句柄可能在线程清理期间被关闭，引发 ValueError。
    本包装器会静默捕获这些异常，确保系统稳定性。
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """Agent 迭代预算计数器（线程安全）。

    每个 Agent（父级或子代理）都有独立的 ``IterationBudget``。
    父级预算上限由 ``max_iterations`` 定义（默认 90）。
    子代理拥有独立预算（默认 50），这意味着父子代理的总迭代次数可能超过父级上限。
    
    ``execute_code``（编程工具调用）的迭代次数通过 :meth:`refund` 返还，不计入预算消耗。
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """尝试消耗一次迭代次数。允许则返回 True。"""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """返还一次迭代次数（例如用于 execute_code 回合）。"""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        """获取剩余迭代次数。"""
        with self._lock:
            return max(0, self.max_total - self._used)


# 绝对不能并发的工具（交互式/面向用户）。
# 当这些工具出现在批次中时，回退到顺序执行。
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# 只读工具，无共享可变会话状态。
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

# 文件工具可在目标路径独立时并发运行。
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# 并发工具执行的最大工作线程数。
_MAX_TOOL_WORKERS = 8

# 匹配可能修改/删除文件的终端命令模式。
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """启发式判断：该终端命令是否包含修改或删除文件的操作？"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """判断当前工具调用批次是否可以安全地并发执行。"""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """获取路径作用域工具的规范化目标文件路径。"""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """判断两个路径是否存在重叠（是否指向同一子树）。"""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

_BUDGET_WARNING_RE = re.compile(
    r"\[BUDGET(?:\s+WARNING)?:\s+Iteration\s+\d+/\d+\..*?\]",
    re.DOTALL,
)


def _sanitize_surrogates(text: str) -> str:
    """将文本中孤立的代理对（Surrogates）替换为 U+FFFD。"""
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_messages_surrogates(messages: list) -> bool:
    """对消息列表中的所有字符串内容执行代理对清洗。"""
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
    return found


def _strip_budget_warnings_from_history(messages: list) -> None:
    """从消息历史中剥离迭代预算警告（Turn 级信号）。"""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or "_budget_warning" not in content and "[BUDGET" not in content:
            continue

        # 优先尝试 JSON（常见情况：dict 中的 _budget_warning 键）
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "_budget_warning" in parsed:
                del parsed["_budget_warning"]
                msg["content"] = json.dumps(parsed, ensure_ascii=False)
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        # 回退：从纯文本工具结果中剥离文本模式
        cleaned = _BUDGET_WARNING_RE.sub("", content).strip()
        if cleaned != content:
            msg["content"] = cleaned


# =========================================================================
# =========================================================================


class AIAgent:
    """
    具备工具调用能力的 AI Agent 编排内核。

    负责管理对话流（主循环）、工具执行以及 AI 模型的响应处理。
    """

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
        pass_session_id: bool = False,
        persist_session: bool = True,
    ):
        """
        初始化 AI Agent。

        AIAgent.__init__() 完成以下五件事：
        1. 选择 provider / API mode / client 行为（见 run_agent.py:553-829）
        2. 基于 toolset 和 check_fn 建立本会话真正可见的工具面（见 run_agent.py:855-883）
        3. 绑定会话落盘状态，包括 session row 和 _last_flushed_db_idx（见 run_agent.py:899-947）
        4. 装配内置记忆与外部 memory provider（见 run_agent.py:969-1078）
        5. 建立 ContextCompressor 与 token/cost/fallback 等会话级状态（见 run_agent.py:1132-1218）
        """
        _install_safe_stdio()

        # ================================================================
        # 第一部分：基础配置与迭代预算
        # ================================================================
        self.model = model
        self.max_iterations = max_iterations
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform
        self._user_id = user_id
        self._print_fn = None
        self.background_review_callback = None
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.persist_session = persist_session
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])

        # ================================================================
        # 【1】选择 provider / API mode / client 行为
        # 根据 base_url 和 provider 自动推断 API 模式：
        # - anthropic_messages: 原生 Anthropic API 或 /anthropic 结尾的兼容端点
        # - codex_responses: OpenAI Codex / 直接 OpenAI URL（GPT-5 等）
        # - chat_completions: 默认 OpenAI 兼容模式
        # ================================================================
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and "chatgpt.com/backend-api/codex" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif self.provider == "anthropic" or (provider_name is None and "api.anthropic.com" in self._base_url_lower):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # 第三方 Anthropic 兼容端点（如 MiniMax、DashScope）使用 /anthropic 后缀
            self.api_mode = "anthropic_messages"
        else:
            self.api_mode = "chat_completions"

        # 直接 OpenAI 会话使用 Responses API 路径（GPT-5.x 工具调用在 /v1/chat/completions 被拒绝）
        if self.api_mode == "chat_completions" and self._is_direct_openai_url():
            self.api_mode = "codex_responses"

        # 预热 OpenRouter 模型元数据缓存（后台线程，避免首次 API 响应时阻塞）
        if self.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self._reasoning_deltas_fired = False  # 由 _fire_reasoning_delta 设置，每次 API 调用重置
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback

        
        # 工具执行状态 — 即使流消费者已注册也能在工具执行期间使用 _vprint（此时无令牌流）
        self._executing_tools = False

        # 中断机制，用于跳出工具循环
        self._interrupt_requested = False
        self._interrupt_message = None  # 触发中断的可选消息
        self._client_lock = threading.RLock()
        
        # 子代理委托状态
        self._delegate_depth = 0        # 0 = 顶级代理，为子代理递增
        self._active_children = []      # 运行中的子代理（用于中断传播）
        self._active_children_lock = threading.Lock()
        
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection

        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # 模型响应配置
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = reasoning_config  # None = use default (medium for OpenRouter)
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns
        
        # Anthropic 提示缓存：通过 OpenRouter 为 Claude 模型自动启用。
        # 通过缓存对话前缀，将多轮对话的输入成本降低约 75%。使用 system_and_3 策略（4 个断点）。
        is_openrouter = self._is_openrouter_url()
        is_claude = "claude" in self.model.lower()
        is_native_anthropic = self.api_mode == "anthropic_messages"
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic
        self._cache_ttl = "5m"  # 默认 5 分钟 TTL（1.25x 写入成本）
        
        # 迭代预算压力：在接近 max_iterations 时警告 LLM。
        # 警告注入到最后一次工具结果的 JSON 中（而非单独消息），以避免破坏消息结构或使缓存失效。
        self._budget_caution_threshold = 0.7   # 70% — 开始收尾的提示
        self._budget_warning_threshold = 0.9   # 90% — 紧急，立即响应
        self._budget_pressure_enabled = True

        # 上下文压力警告：当上下文满时通知用户（非 LLM）。纯信息性——显示在 CLI 输出中，
        # 并通过 gateway 平台的 status_callback 发送。不会注入到消息中。
        self._context_pressure_warned = False

        # 活动跟踪——在每次 API 调用、工具执行和流块时更新。
        # 用于 gateway 超时处理器报告 agent 被杀死时正在做什么，以及"仍在工作"通知以显示进度。
        self._last_activity_ts: float = time.time()
        self._last_activity_desc: str = "initializing"
        self._current_tool: str | None = None
        self._api_call_count: int = 0

        # 集中式日志——agent.log (INFO+) 和 errors.log (WARNING+) 都位于 ~/.hermes/logs/ 下。
        # 幂等的，所以 gateway 模式（每条消息创建一个新 AIAgent）不会重复处理器。
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            setup_verbose_logging()
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            if self.quiet_mode:
                # 安静模式（CLI 默认）：抑制控制台上的所有工具/基础设施日志噪音。
                # TUI 有自己的丰富状态显示；logger INFO/WARNING 消息只会添乱。
                # 文件处理器（agent.log, errors.log）仍然捕获所有内容。
                for quiet_logger in [
                    'tools',               # 所有工具模块（终端、浏览器、Web、文件等）
                    'run_agent',            # Agent 运行器内部逻辑
                    'trajectory_compressor',
                    'cron',                 # 调度器（仅与守护进程模式相关）
                    'hermes_cli',           # CLI 辅助函数
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # 内部流回调（流式 TTS 期间设置）。
        # 在此初始化以便 _vprint 在 run_conversation 之前可以引用它。
        self._stream_callback = None
        self._stream_needs_break = False
        self._persist_user_message_idx = None
        self._persist_user_message_override = None  # 用户消息覆盖
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # ================================================================
        # 【1续】初始化 LLM client
        # 根据 api_mode 选择不同的客户端初始化方式
        # ================================================================
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        if self.api_mode == "anthropic_messages":
            # Anthropic 原生模式：使用 Anthropic SDK
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            _is_native_anthropic = self.provider == "anthropic"
            effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            self._is_anthropic_oauth = _is_oat(effective_key)
            self._anthropic_client = build_anthropic_client(effective_key, base_url)
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                if effective_key and len(effective_key) > 12:
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        else:
            # OpenAI 兼容模式：通过集中式 provider router 解析凭证
            if api_key and base_url:
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args
                effective_base = base_url
                if "openrouter" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.githubcopilot.com" in effective_base.lower():
                    from hermes_cli.models import copilot_default_headers
                    client_kwargs["default_headers"] = copilot_default_headers()
                elif "api.kimi.com" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.3",
                    }
            else:
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_explicit.upper()}_API_KEY environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    client_kwargs = {
                        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                        "base_url": OPENROUTER_BASE_URL,
                        "default_headers": {
                            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                            "X-OpenRouter-Title": "Hermes Agent",
                            "X-OpenRouter-Categories": "productivity,cli-agent",
                        },
                    }

            self._client_kwargs = client_kwargs

            # Claude on OpenRouter：启用细粒度工具流式输出，防止模型思考时连接超时
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if "openrouter" in _effective_base and "claude" in (self.model or "").lower():
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            try:
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        # ================================================================
        # 【1续】Provider fallback chain
        # 当 primary provider 耗尽（限流、过载、连接失败）时按序尝试备用 provider
        # 支持 fallback_model（单个 dict）或 fallback_model（list）格式
        # ================================================================
        if isinstance(fallback_model, list):
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            self._fallback_chain = [fallback_model]
        else:
            self._fallback_chain = []
        self._fallback_index = 0
        self._fallback_activated = False
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 Fallback chain ({len(self._fallback_chain)} providers): " +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # ================================================================
        # 【2】基于 toolset 和 check_fn 建立本会话真正可见的工具面
        # get_tool_definitions() 会根据 enabled/disabled toolsets 过滤工具
        # ================================================================
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )

        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")

        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")

        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")

        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")

        if self._use_prompt_caching and not self.quiet_mode:
            source = "native Anthropic" if is_native_anthropic else "Claude via OpenRouter"
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")

        # ================================================================
        # 【3】绑定会话落盘状态
        # - 生成或复用 session_id
        # - 设置会话日志路径
        # - 初始化 SQLite session store（用于会话持久化和搜索）
        # - _last_flushed_db_idx 追踪 DB 写入游标，防止重复写入
        # ================================================================
        self.session_start = datetime.now()
        if session_id:
            self.session_id = session_id
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"

        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"

        self._session_messages: List[Dict[str, Any]] = []
        self._cached_system_prompt: Optional[str] = None

        # 文件系统检查点管理器（非工具，对用户透明）
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )

        # SQLite session store
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # 追踪 DB 写入游标，防止重复写入
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                    parent_session_id=self._parent_session_id,
                )
            except Exception as e:
                # 瞬时 SQLite 锁争用（如 CLI 和 gateway 并发写入）不应永久禁用此 agent 的 session_search。
                # 保持 _session_db 活跃——一旦锁清除，后续消息刷新和 session_search 调用仍可工作。
                # session 行可能在此运行中缺失于索引，但可恢复（刷新会 upsert 行）。
                logger.warning(
                    "Session DB create_session failed (session_search still available): %s", e
                )
        
        # 用于任务规划的内存待办事项列表（每个 Agent/会话一个）
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # 为记忆、技能和压缩部分加载一次配置
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}

        # ================================================================
        # 【4】装配内置记忆与外部 memory provider
        # 内置记忆：MEMORY.md + USER.md，从磁盘加载
        # 外部记忆：memory.provider 配置的插件（如 honcho）
        # ================================================================
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass

        # 外部 memory provider 插件（同时只能有一个）
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                # 自动迁移：如果 Honcho 被积极配置（启用 + 凭证）但 memory.provider 未设置，
                # 则自动激活 honcho 插件。仅拥有配置文件是不够的——用户可能已禁用 Honcho，
                # 或该文件可能来自其他工具。
                if not _mem_provider_name:
                    try:
                        from plugins.memory.honcho.client import HonchoClientConfig as _HCC
                        _hcfg = _HCC.from_global_config()
                        if _hcfg.enabled and (_hcfg.api_key or _hcfg.base_url):
                            _mem_provider_name = "honcho"
                            # 持久化，以便只自动迁移一次
                            try:
                                from hermes_cli.config import load_config as _lc, save_config as _sc
                                _cfg = _lc()
                                _cfg.setdefault("memory", {})["provider"] = "honcho"
                                _sc(_cfg)
                            except Exception:
                                pass
                            if not self.quiet_mode:
                                print("  ✓ Auto-migrated Honcho to memory provider plugin.")
                                print("    Your config and data are preserved.\n")
                    except Exception:
                        pass

                if _mem_provider_name:
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        from hermes_constants import get_hermes_home as _ghh
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(_ghh()),
                            "agent_context": "primary",
                        }
                        # 为每个用户内存作用域线程化 gateway 用户身份
                        if self._user_id:
                            _init_kwargs["user_id"] = self._user_id
                        # 每个 profile provider 作用域的 profile 身份
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("Memory provider '%s' activated", _mem_provider_name)
                    else:
                        logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("Memory provider plugin init failed: %s", _mpe)
                self._memory_manager = None

        # 将记忆提供商工具 Schema 注入到工具面
        if self._memory_manager and self.tools is not None:
            for _schema in self._memory_manager.get_all_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)

        # 技能配置：技能创建提醒的间隔
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
        except Exception:
            pass

        # 工具调用强制配置："auto"（默认 - 匹配硬编码的模型列表）、true（始终）、false（从不）或子字符串列表。
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # ================================================================
        # 【5】建立 ContextCompressor 与 token/cost/fallback 等会话级状态
        # ContextCompressor：自动上下文管理，当接近模型 context limit 时压缩会话
        # 配置通过 config.yaml（compression 部分）
        # ================================================================
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
        compression_summary_model = _compression_cfg.get("summary_model") or None
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))

        # Read explicit context_length override from model config
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                _config_context_length = None

        # Check custom_providers per-model context_length
        if _config_context_length is None:
            _custom_providers = _agent_cfg.get("custom_providers")
            if isinstance(_custom_providers, list):
                for _cp_entry in _custom_providers:
                    if not isinstance(_cp_entry, dict):
                        continue
                    _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                    if _cp_url and _cp_url == self.base_url.rstrip("/"):
                        _cp_models = _cp_entry.get("models", {})
                        if isinstance(_cp_models, dict):
                            _cp_model_cfg = _cp_models.get(self.model, {})
                            if isinstance(_cp_model_cfg, dict):
                                _cp_ctx = _cp_model_cfg.get("context_length")
                                if _cp_ctx is not None:
                                    try:
                                        _config_context_length = int(_cp_ctx)
                                    except (TypeError, ValueError):
                                        pass
                        break
        
        self.context_compressor = ContextCompressor(
            model=self.model,
            threshold_percent=compression_threshold,
            protect_first_n=3,
            protect_last_n=compression_protect_last,
            summary_target_ratio=compression_target_ratio,
            summary_model_override=compression_summary_model,
            quiet_mode=self.quiet_mode,
            base_url=self.base_url,
            api_key=getattr(self, "api_key", ""),
            config_context_length=_config_context_length,
            provider=self.provider,
        )
        self.compression_enabled = compression_enabled
        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0

        # Cumulative token usage for the session
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # ── Ollama num_ctx 注入 ──
        # Ollama 默认使用 2048 上下文，无论模型能力如何。
        # 当连接到 Ollama 服务器时，检测模型的最大上下文并在每个聊天请求中传递 num_ctx，以便使用完整的窗口。
        # 用户覆盖：在 config.yaml 中设置 model.ollama_num_ctx 以限制 VRAM 使用。
        self._ollama_num_ctx: int | None = None
        _ollama_num_ctx_override = None
        if isinstance(_model_cfg, dict):
            _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
        if _ollama_num_ctx_override is not None:
            try:
                self._ollama_num_ctx = int(_ollama_num_ctx_override)
            except (TypeError, ValueError):
                logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
        if self._ollama_num_ctx is None and self.base_url and is_local_endpoint(self.base_url):
            try:
                _detected = query_ollama_num_ctx(self.model, self.base_url)
                if _detected and _detected > 0:
                    self._ollama_num_ctx = _detected
            except Exception as exc:
                logger.debug("Ollama num_ctx detection failed: %s", exc)
        if self._ollama_num_ctx and not self.quiet_mode:
            logger.info(
                "Ollama num_ctx: will request %d tokens (model max from /api/show)",
                self._ollama_num_ctx,
            )

        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

        # ================================================================
        # 【5续】快照 primary runtime，供每轮恢复
        # 当 fallback 在某轮激活后，下一轮会恢复这些值，使首选模型获得新的尝试机会
        # ================================================================
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            # Compressor state that _try_activate_fallback() overwrites
            "compressor_model": _cc.model,
            "compressor_base_url": _cc.base_url,
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": _cc.provider,
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

    def reset_session_state(self):
        """重置会话级状态（Token 计数器、成本统计等）。
        
        此方法封装了所有会话指标的重置逻辑，包括：
        - Token 使用量（输入、输出、总计、Prompt、Completion）
        - 缓存读写 Token
        - API 调用次数
        - 推理 Token
        - 预估成本
        - 上下文压缩器的内部计数器
        """
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        self._user_turn_count = 0

        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.last_prompt_tokens = 0
            self.context_compressor.last_completion_tokens = 0
            self.context_compressor.last_total_tokens = 0
            self.context_compressor.compression_count = 0
            self.context_compressor._context_probed = False
            self.context_compressor._context_probe_persistable = False
            self.context_compressor._previous_summary = None
    
    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """在运行时切换模型/供应商 (Provider)。

        由 /model 命令调用。该方法执行实际的运行时切换：
        重建客户端、更新缓存标志并刷新上下文压缩器。
        同时更新 ``_primary_runtime`` 确保变更跨 Turn 持久生效。
        """
        import logging
        from hermes_cli.providers import determine_api_mode

        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        old_model = self.model
        old_provider = self.provider

        self.model = new_model
        self.provider = new_provider
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode
        if api_key:
            self.api_key = api_key

        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            effective_key = api_key or self.api_key or resolve_anthropic_token() or ""
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key)
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        is_native_anthropic = api_mode == "anthropic_messages"
        self._use_prompt_caching = (
            ("openrouter" in (self.base_url or "").lower() and "claude" in new_model.lower())
            or is_native_anthropic
        )

        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
            )
            self.context_compressor.model = self.model
            self.context_compressor.base_url = self.base_url
            self.context_compressor.api_key = self.api_key
            self.context_compressor.provider = self.provider
            self.context_compressor.context_length = new_context_length
            self.context_compressor.threshold_tokens = int(
                new_context_length * self.context_compressor.threshold_percent
            )

        self._cached_system_prompt = None

        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "compressor_model": _cc.model if _cc else self.model,
            "compressor_base_url": _cc.base_url if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": _cc.provider if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        self._fallback_activated = False
        self._fallback_index = 0

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )

    def _safe_print(self, *args, **kwargs):
        """线程/环境安全的打印方法，处理断开管道或已关闭的 stdout。
        
        在 headless 环境（systemd, Docker）中，stdout 可能变得不可用。
        此方法静默处理 OSError，防止内核崩溃。
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """详细打印方法。在流式输出 Token 时默认静默（除非 force=True）。
        
        在工具执行期间允许打印。
        在响应完成后的收尾阶段（_mute_post_response）静默非强制输出。
        """
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """判断是否允许启动加载动画 (Spinner)。
        
        在 headless/stdio 协议环境中，未经重定向的动画可能会污染协议流（如 ACP JSON-RPC）。
        仅当输出已重定向或处于真实 TTY 时允许。
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _emit_status(self, message: str) -> None:
        """向 CLI 和 Gateway 通道发布生命周期状态消息。
        
        CLI 用户将通过 _vprint(force=True) 看到消息。
        Gateway 消费者通过 status_callback 接收。
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """当基准 URL 指向 OpenAI 原生 API 时返回 True。"""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _is_openrouter_url(self) -> bool:
        """当基准 URL 指向 OpenRouter 时返回 True。"""
        return "openrouter" in self._base_url_lower

    def _max_tokens_param(self, value: int) -> dict:
        """根据不同的 Provider 返回正确的最大 Token 参数名称。
        
        新版 OpenAI 模型使用 'max_completion_tokens'，其余使用 'max_tokens'。
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """检查在推理/思考块（think block）之后是否包含实际文本内容。
        
        用于检测模型是否仅输出了推理而未给出最终响应（此时应重试）。
        """
        if not content:
            return False

        cleaned = self._strip_think_blocks(content)

        return bool(cleaned.strip())
    
    def _strip_think_blocks(self, content: str) -> str:
        """移除内容中的推理/思考块，仅保留可见文本。"""
        if not content:
            return ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL)
        content = re.sub(r'</?(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>\s*', '', content, flags=re.IGNORECASE)
        return content

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """检测应继续而非结束本轮的计划/确认消息。"""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action
    
    
    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        从 assistant 消息中提取 reasoning/thinking 内容。

        OpenRouter 和各种 provider 可以用多种格式返回 reasoning：
        1. message.reasoning - 直接 reasoning 字段（DeepSeek、Qwen 等）
        2. message.reasoning_content - 替代字段（Moonshot AI、Novita 等）
        3. message.reasoning_details - {type, summary, ...} 对象数组（OpenRouter 统一格式）

        Args:
            assistant_message: API 响应的 assistant 消息对象

        Returns:
            组合的 reasoning 文本，若无 reasoning 则返回 None
        """
        reasoning_parts = []
        
        # 检查直接 reasoning 字段
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # 检查 reasoning_content 字段（某些 provider 使用的替代名称）
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # 如果与 reasoning 相同则不重复
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # 检查 reasoning_details 数组（OpenRouter 统一格式）
        # 格式：[{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # 从推理详情对象中提取摘要
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # 某些 provider 将 reasoning 直接嵌入 assistant content，
        # 而非返回结构化 reasoning 字段。仅在未找到结构化 reasoning 时
        # 回退到内联提取。
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # 组合所有推理部分
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None

    def _cleanup_task_resources(self, task_id: str) -> None:
        """清理给定任务的 VM 和浏览器资源。"""
        try:
            cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # 后台记忆/技能复盘
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
        "Focus on: was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome?\n\n"
        "If a relevant skill already exists, update it with what you learned. "
        "Otherwise, create a new skill if the approach is reusable.\n"
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves — their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "If so, save using the memory tool.\n\n"
        "**Skills**: Was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome? If a relevant skill "
        "already exists, update it. Otherwise, create a new one if the approach is reusable.\n\n"
        "Only act if there's something genuinely worth saving. "
        "If nothing stands out, just say 'Nothing to save.' and stop."
    )

    # ================================================================
    # 【文档锚点 5A】后台复盘
    # 1. AIAgent 在会话内维护 memory / skill nudge 计数器
    # 2. 当 turn 完成且满足条件时，run_conversation() 会在响应交付后启动 _spawn_background_review()
    #    （见 run_agent.py:9146-9172）
    # 3. _spawn_background_review() 会 fork 一个安静的 AIAgent，读取本轮 messages_snapshot，
    #    判断是否该写 memory 或更新 / 创建 skill（见 run_agent.py:1718-1822）
    # ================================================================
    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """启动后台线程来复盘对话并决定是否保存 memory/skill

        创建一个完整的 AIAgent fork，使用与主会话相同的 model、tools 和 context。
        复盘 prompt 作为下一个 user turn 添加到 fork 的对话中。
        直接写入共享的 memory/skill store，不修改主会话历史，也不产生用户可见的输出。
        """
        # 【文档锚点 5A】主任务结束后异步 fork 一个安静 agent 做经验沉淀
        import threading

        # 根据触发的条件选择合适的复盘 Prompt
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib, os as _os
            review_agent = None
            try:
                with open(_os.devnull, "w") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # 扫描复盘 agent 的消息以获取成功的工具操作，
                # 并向用户显示紧凑摘要。
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (target and "add" in message.lower()):
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "Entry added" in message:
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(f"💾 {summary}")
                        except Exception:
                            pass

            except Exception as e:
                logger.debug("Background memory/skill review failed: %s", e)
            finally:
                # 显式关闭 OpenAI/httpx 客户端，以免 GC 在已死的 asyncio 事件循环上尝试
                # 清理（这会在终端产生"Event loop is closed"错误）。
                if review_agent is not None:
                    client = getattr(review_agent, "client", None)
                    if client is not None:
                        try:
                            review_agent._close_openai_client(
                                client, reason="bg_review_done", shared=True
                            )
                            review_agent.client = None
                        except Exception:
                            pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """在持久化/返回前重写当前轮次的用户消息。

        某些调用路径需要仅 API 的用户消息变体，而不让该合成文本泄露到
        持久化记录或恢复的会话历史中。当为当前轮次配置了覆盖时，
        就地修改内存中的消息列表，以便持久化和返回的历史都保持干净。
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """在任何退出路径上都将会话状态保存到 JSON 日志和 SQLite。

        确保对话永不丢失，即使在错误或提前返回时也是如此。
        当 ``persist_session=False`` 时跳过（临时辅助流程）。
        """
        if not self.persist_session:
            return
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """将任何未刷新的消息持久化到 SQLite 会话存储。

        使用 _last_flushed_db_idx 跟踪已写入的消息，以便重复调用
        （来自多个退出路径）仅写入真正的新消息——防止重复写入 bug (#860)。
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # 如果 create_session() 在启动时失败（如瞬时锁），
            # session 行可能尚不存在。ensure_session() 使用 INSERT OR IGNORE，
            # 所以当行已存在时它是一个空操作。
            self._session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        获取到（但不包含）最后一个 assistant 轮次的消息。

        当我们需要"回滚"到对话中最后一个成功点时使用此方法，
        通常是当最后一个 assistant 消息不完整或格式错误时。

        Args:
            messages: 完整消息列表

        Returns:
            到最后一个完整 assistant 轮次的消息（以 user/tool 消息结束）
        """
        if not messages:
            return []
        
        # 查找最后一个 assistant 消息的索引
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # 未找到 assistant 消息，返回全部
            return messages.copy()
        
        # 返回最后一个 assistant 消息之前的所有内容（以 user/tool 结束）
        return messages[:last_assistant_idx]
    
    def _format_tools_for_system_message(self) -> str:
        """
        将工具定义格式化为 system prompt 中使用的 JSON 字符串（轨迹格式）。
        """
        if not self.tools:
            return "[]"
        
        # 转换工具定义为轨迹（Trajectory）所需的格式
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    
    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        将内部消息格式转换为用于保存的轨迹（Trajectory）格式。
        """
        trajectory = []
        
        # 添加带工具定义的 System Message
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # 将原始用户查询作为第一个 Human 消息
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # 跳过第一条（因为已添加），prefill 消息在 API 调用时注入，不计入消息列表
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    # 格式化带工具调用的 Assistant 消息
                    content = ""
                    
                    # 如果有原生思维链内容，包裹在 <think> 标签中
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # 将 <REASONING_SCRATCHPAD> 标签转换为 <think>
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # 添加带 XML 标签包装的工具调用
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # 确保每个 GPT 轮次都有 <think> 块，保持格式一致性
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # 收集后续所有工具响应
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        tool_response = "<tool_response>\n"
                        
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # 合并工具响应并添加
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1
                
                else:
                    # 不带工具调用的普通 Assistant 消息
                    content = ""
                    
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    
    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        将对话轨迹保存到 JSONL 文件。
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)
    
    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """从 API 错误中提取人类可读的单行信息。

        通过提取 <title> 标签（而非转储原始 HTML）来处理 Cloudflare HTML 错误页面（502、503 等）。
        其他情况回退到截断的 str(error)。
        """
        import re as _re
        raw = str(error)

        # Cloudflare / 代理 HTML 页面：抓取 <title> 以获得干净的摘要
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = _re.search(r"<title[^>]*>([^<]+)</title>", raw, _re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # 如果存在，也抓取 Cloudflare Ray ID
            ray = _re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # 来自 OpenAI/Anthropic SDK 的 JSON 体错误
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # 回退：截断原始字符串，但提供比 200 字符更多的空间
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """
        清理错误消息以供用户显示，移除 HTML 内容并截断。

        Args:
            error_msg: API 或异常的原始错误消息

        Returns:
            干净的、用户友好的错误消息
        """
        if not error_msg:
            return "Unknown error"
            
        # Remove HTML content (common with CloudFlare and gateway error pages)
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "Service temporarily unavailable (HTML error page returned)"
            
        # Remove newlines and excessive whitespace
        cleaned = ' '.join(error_msg.split())
        
        # Truncate if too long
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
            
        return cleaned

    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """从 Provider 错误中提取结构化的频率限制（Rate-limit）详情。"""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """为 ``post_api_request`` 插件提供 Token 使用量统计摘要。"""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        导出用于调试的 HTTP 请求记录。
        
        捕获 api_kwargs 中的请求体（排除超时等传输层参数）。用于排查 Provider 侧的 4xx 错误。
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        将完整的原始会话保存到 JSON 文件中。

        存储 Agent 看到的每一条确切消息：用户消息、
        Assistant 消息（带推理、完成原因、工具调用）、
        工具响应（带 tool_call_id、tool_name）以及注入的系统
        消息（压缩摘要、待办事项快照等）。

        REASONING_SCRATCHPAD 标签被转换为 <think> 块以保持一致性。
        每轮之后重写，以便始终反映最新状态。
        """
        # 【文档锚点 4B】JSON session log：保留 agent 视角下的完整原始消息快照
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # 清理 Assistant 内容以用于会话日志
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # 保护：永远不要用消息较少的会话日志覆盖较大的会话日志。
            # 这可以防止在 --resume 加载其消息尚未完全写入 SQLite 的会话时发生数据丢失
            # ——恢复后的 Agent 从部分历史开始，否则会破坏完整的 JSON 日志。
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # 文件损坏 — 允许覆盖

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    
    def interrupt(self, message: str = None) -> None:
        """
        请求 Agent 中断其当前的工具调用循环。
        
        从另一个线程调用此方法（例如输入处理器、消息接收器）
        以优雅地停止 Agent 并处理新消息。
        
        同时向运行时间较长的工具执行（例如终端命令）发出信号
        提前终止，以便 Agent 能够立即响应。
        
        Args:
            message: 触发中断的可选新消息。
                     如果提供，Agent 将在响应上下文中包含此内容。
        
        示例 (CLI):
            # 在单独的输入线程中：
            if user_typed_something:
                agent.interrupt(user_input)
        
        示例 (消息平台):
            # 当活跃会话收到新消息时：
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # 立即通知所有工具中止任何正在进行的操作
        _set_interrupt(True)
        # 将中断传播到任何运行中的子代理（子代理委托）
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """清除任何挂起的中断请求和全局工具中断信号。"""
        self._interrupt_requested = False
        self._interrupt_message = None
        _set_interrupt(False)

    def _touch_activity(self, desc: str) -> None:
        """更新最后活动的时间戳和描述（线程安全）。"""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc

    def get_activity_summary(self) -> dict:
        """返回 Agent 当前活动的快照，用于诊断。

        由 gateway 超时处理器调用，用于报告 Agent 被杀掉时正在做什么，
        以及通过定期的"仍在工作"通知。
        """
        elapsed = time.time() - self._last_activity_ts
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
        }

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """关闭记忆提供商——在实际会话边界处调用。

        这会调用记忆管理器的 on_session_end() 然后是 shutdown_all()。
        不是每轮都调用——仅在 CLI 退出、/reset、gateway 会话过期等情况下调用。
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
    
    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        从对话历史中恢复待办事项状态。
        
        Gateway 为每条消息创建一个全新的 AIAgent，因此内存中的
        TodoStore 是空的。我们扫描历史以寻找最近的 todo
        工具响应并重新回放它以重建状态。
        """
        # 向后遍历历史以找到最近的 todo 工具响应
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # 快速检查：todo 响应包含 "todos" 键
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # 将项重新回放进存储（替换模式）
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """检查是否已请求中断。"""
        return self._interrupt_requested










    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        组装完整的 system prompt（分层的）

        真正的 prompt 构建逻辑（按顺序）：
        1. 身份层：SOUL/default identity（见 run_agent.py:2599-2610）
        2. 工具行为约束：memory / session_search / skill_manage guidance（见 run_agent.py:2611-2621）
        3. 内置记忆与外部 memory provider 的 system block（见 run_agent.py:2665-2683）
        4. 技能索引层（见 run_agent.py:2685-2701）
        5. 项目上下文层（见 agent/prompt_builder.py:944-983）

        注意：ephemeral_system_prompt 不在这里，它在 API 调用时才注入
        """
        # 【文档锚点 3A】身份层：SOUL.md 或默认 identity
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # 【文档锚点 3A】工具行为约束层：根据加载的工具注入对应的 guidance
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
        # Tool-use enforcement: tells the model to actually call tools instead
        # of describing intended actions.  Controlled by config.yaml
        # agent.tool_use_enforcement:
        #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
        #   true  — always inject (all models)
        #   false — never inject
        #   list  — custom model-name substrings to match
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # so it can refer the user to them rather than reinventing answers.

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # 【文档锚点 3A】内置记忆与外部 memory provider 的 system block
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        # 【文档锚点 3A】技能索引层
        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        # 【文档锚点 3A】项目上下文层
        if not self.skip_context_files:
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(prompt_parts)

    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)
    # =========================================================================

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """在每次 LLM 调用前修复孤立的 tool_call / tool_result 对。

        无条件运行——不依赖于上下文压缩器是否存在——因此
        会话加载或手动消息操作中的孤立项始终被捕获。
        """
        # --- 角色白名单：删除 API 不接受的角色的消息 ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in AIAgent._VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. 删除没有匹配 assistant 调用的工具结果
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. 为结果被丢弃的调用注入桩结果
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """将多余的 delegate_task 调用截断到 MAX_CONCURRENT_CHILDREN。

        delegate_tool 在单次调用中限制任务列表，但模型可以在一轮中
        发出多个独立的 delegate_task tool_calls。这会截断多余的调用，
        保留所有非委托调用。

        如果不需要截断则返回原始列表。
        """
        from tools.delegate_tool import MAX_CONCURRENT_CHILDREN
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= MAX_CONCURRENT_CHILDREN:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < MAX_CONCURRENT_CHILDREN:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "MAX_CONCURRENT_CHILDREN=%d limit",
            delegate_count - MAX_CONCURRENT_CHILDREN, MAX_CONCURRENT_CHILDREN,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """删除单轮中重复的 (tool_name, arguments) 对。

        只保留每个唯一对的第一个出现。
        如果未找到重复则返回原始列表。
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """在中止前尝试修复不匹配的工具名称。

        1. 尝试小写
        2. 尝试标准化（小写 + 连字符/空格 -> 下划线）
        3. 尝试模糊匹配（difflib，cutoff=0.7）

        如果在 valid_tool_names 中找到返回修复后的名称，否则返回 None。
        """
        from difflib import get_close_matches

        # 1. Lowercase
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. Normalize
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. Fuzzy match
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """
        使缓存的 system prompt 失效，强制在下一轮重新构建。

        在上下文压缩事件后调用。同时从磁盘重新加载内存，
        以便重建的 prompt 捕获此会话的任何写入。
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """将 chat-completions 工具模式转换为 Responses function-tool 模式。"""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """从工具调用内容生成确定性 call_id。

        用作 API 未提供 call_id 时的回退。
        确定性 ID 防止缓存失效——随机 UUID 会使每个 API 调用的前缀唯一，
        破坏 OpenAI 的提示缓存。
        """
        import hashlib
        seed = f"{fn_name}:{arguments}:{index}"
        digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"call_{digest}"

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """将存储的工具 ID 拆分为 (call_id, response_item_id)。"""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """构建有效的 Responses `function_call.id`（必须以 `fc_` 开头）。"""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将内部聊天风格消息转换为 Responses 输入项。"""
        items: List[Dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # 重放上一轮的加密 reasoning 项，
                    # 以便 API 保持连贯的 reasoning 链。
                    codex_reasoning = msg.get("codex_reasoning_items")
                    has_codex_reasoning = False
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                items.append(ri)
                                has_codex_reasoning = True

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})
                    elif has_codex_reasoning:
                        # Responses API 要求在每个 reasoning 项之后有一个后续项
                        #（否则：missing_following_item 错误）。
                        # 当 assistant 只产生 reasoning 而没有可见内容时，
                        # 发出一个空的 assistant 消息作为所需的后续项。
                        items.append({"role": "assistant", "content": ""})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(
                                tc.get("id")
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                                else:
                                    _raw_args = str(fn.get("arguments", "{}"))
                                    call_id = self._deterministic_call_id(fn_name, _raw_args, len(items))
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": arguments,
                            })
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })

        return items

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        reasoning_item["id"] = item_id
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex Responses request 'model' must be a non-empty string.")
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Codex Responses request 'tools' must be a list when provided.")
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model", "instructions", "input", "tools", "store",
            "reasoning", "include", "max_output_tokens", "temperature",
            "tool_choice", "parallel_tool_calls", "prompt_cache_key",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "store": False,
        }
        if normalized_tools is not None:
            normalized["tools"] = normalized_tools

        # 透传推理配置
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include

        # 透传最大输出 token 数和温度参数
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # 透传工具选择、并行工具调用及提示词缓存键
        for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

        unexpected = sorted(key for key in api_kwargs if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """从 Responses 消息输出项中提取 Assistant 文本。"""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """从 Responses 推理项中提取紧凑的推理文本。"""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
        """将 Responses API 对象标准化为类似 assistant_message 的对象。"""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            # 当通过流式事件交付答案时，Codex 后端可能返回空输出。
            # 尝试回退到 output_text。
            out_text = getattr(response, "output_text", None)
            if isinstance(out_text, str) and out_text.strip():
                logger.debug(
                    "Codex response has empty output but output_text is present (%d chars); "
                    "synthesizing output item.", len(out_text.strip()),
                )
                output = [SimpleNamespace(
                    type="message", role="assistant", status="completed",
                    content=[SimpleNamespace(type="output_text", text=out_text.strip())],
                )]
                response.output = output
            else:
                raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                # 捕获完整的推理项以便跨轮次连续。
                # encrypted_content 是 API 在后续轮次中需要的加密数据，用于保持一致的推理链。
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # 捕获摘要 —— 重新播放推理项时 API 需要
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            # 响应仅包含推理（加密的思考状态），没有可见内容或工具调用。
            # 模型仍在思考，需要下一轮来产生实际答案。
            # 处理为 "incomplete" 以便 Codex 持续路径正确处理。
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """检查 OpenAI 客户端是否已关闭。

        处理 is_closed 的属性和方法两种形式：
        - httpx.Client.is_closed 是布尔属性
        - openai.OpenAI.is_closed 是返回布尔值的方法

        先前 bug：getattr(client, "is_closed", False) 返回绑定方法，
        这始终为真，导致每次调用时不必要地重新创建客户端。
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # 处理方法 (openai SDK) 与属性 (httpx) 的差异
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """强制关闭底层 TCP 套接字以防止 CLOSE-WAIT 累积。

        当 provider 在流中间断开连接时，httpx 的 ``client.close()``
        执行优雅关闭，使套接字处于 CLOSE-WAIT 状态直到 OS 超时（通常几分钟）。
        此方法遍历 httpx 传输池并发出 ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()``
        以强制立即 TCP RST，释放文件描述符。

        返回强制关闭的套接字数。
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx 使用 httpcore 连接池；连接根据版本不同存储在 _connections (列表) 或 _pool (列表) 中。
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # 先强制关闭 TCP 套接字以防止 CLOSE-WAIT 累积，
        # 然后执行优雅的 SDK 级关闭。
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """检测并清理主客户端上的死 TCP 连接。

        检查 httpx 连接池中处于不健康状态（CLOSE-WAIT、错误）的套接字。
        如果发现任何死连接，强制关闭所有套接字并从头重建主客户端。

        如果发现并清理了死连接则返回 True。
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # 检查空闲但套接字已关闭的连接
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # 使用非阻塞 recv peek 探测套接字健康状况
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # 无可用数据——套接字健康
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """执行一个流式 Responses API 请求并返回最终响应。"""
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        self._reasoning_deltas_fired = False
        # 累积流式文本，以便在 get_final_response() 返回空输出时恢复
        #（例如 chatgpt.com backend-api 发送 response.incomplete 而非 response.completed）。
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # 触发文本内容增量回调（在工具调用期间抑制）
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # 跟踪工具调用以抑制文本流
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # 触发 reasoning 回调
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                        # 收集已完成的输出项——某些后端
                        #（chatgpt.com/backend-api/codex）通过 response.output_item.done
                        # 流式传输有效项，但 SDK 的 get_final_response() 返回空输出列表。
                        elif event_type == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None:
                                collected_output_items.append(done_item)
                        # 记录未完成的终端事件以供诊断
                        elif event_type in ("response.incomplete", "response.failed"):
                            resp_obj = getattr(event, "response", None)
                            status = getattr(resp_obj, "status", None) if resp_obj else None
                            incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                            logger.warning(
                                "Codex Responses stream received terminal event %s "
                                "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                                event_type, status, incomplete_details,
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # PATCH: ChatGPT Codex 后端流式传输有效输出项，
                    # 但 get_final_response() 可能返回空输出列表。
                    # 从收集的项回填或从增量合成。
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """针对 Codex 风格 Responses 后端流式完成边缘情况的回退路径。"""
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._preflight_codex_api_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # 针对 Mock 或仍返回具体响应的提供商的兼容性适配。
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        collected_output_items: list = []
        collected_text_deltas: list = []
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

                # 收集输出项和文本增量以便回填
                if event_type == "response.output_item.done":
                    done_item = getattr(event, "item", None)
                    if done_item is None and isinstance(event, dict):
                        done_item = event.get("item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif event_type in ("response.output_text.delta",):
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(delta)

                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    # 从收集的流事件中回填空输出
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            terminal_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex fallback stream: backfilled %d output items",
                                len(collected_output_items),
                            )
                        elif collected_text_deltas:
                            assembled = "".join(collected_text_deltas)
                            terminal_response.output = [SimpleNamespace(
                                type="message", role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex fallback stream: synthesized from %d deltas (%d chars)",
                                len(collected_text_deltas), len(assembled),
                            )
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous 请求不应继承仅限 OpenRouter 的归因头。
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # 仅为原生 Anthropic 提供商刷新凭据。
        # 其他 anthropic_messages 提供商（如 MiniMax、阿里巴巴等）使用各自的密钥。
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(new_token, getattr(self, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # 更新 OAuth 标志 —— 令牌类型可能已更改（API 密钥 ↔ OAuth）
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _OR_HEADERS

        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.3"}
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(runtime_key, runtime_base)
            self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """尝试通过凭据池轮换进行凭据恢复。

        返回 (是否恢复, 是否已重试 429)。
        遇到 429：第一次出现时重试同一凭据（将标志设为 True）。
                连续第二次出现 429 时轮换到下一个凭据（重置标志）。
        遇到 402：立即轮换（计费耗尽无法通过重试解决）。
        遇到 401：在轮换前尝试刷新令牌。
        """
        pool = self._credential_pool
        if pool is None or status_code is None:
            return False, has_retried_429

        if status_code == 402:
            next_entry = pool.mark_exhausted_and_rotate(status_code=402, error_context=error_context)
            if next_entry is not None:
                logger.info(f"Credential 402 (billing) — rotated to pool entry {getattr(next_entry, 'id', '?')}")
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if status_code == 429:
            if not has_retried_429:
                return False, True
            next_entry = pool.mark_exhausted_and_rotate(status_code=429, error_context=error_context)
            if next_entry is not None:
                logger.info(f"Credential 429 (rate limit) — rotated to pool entry {getattr(next_entry, 'id', '?')}")
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if status_code == 401:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential 401 — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # 刷新失败 —— 轮换到下一个凭据而不是放弃。
            # 失败的条目已被 try_refresh_current() 标记为耗尽。
            next_entry = pool.mark_exhausted_and_rotate(status_code=401, error_context=error_context)
            if next_entry is not None:
                logger.info(f"Credential 401 (refresh failed) — rotated to pool entry {getattr(next_entry, 'id', '?')}")
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        在后台线程中运行 API 调用，以便主对话循环
        可以在不等待完整 HTTP 往返的情况下检测中断。

        每个工作线程获得自己的 OpenAI 客户端实例。中断只
        关闭该工作线程本地的客户端，因此重试和其他请求永远不会
        继承已关闭的传输。
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(reason="codex_stream_request")
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(reason="chat_completion_request")
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                # 强制关闭正在进行的线程本地 HTTP 连接以停止 token 生成，
                # 同时避免污染用于后续重试的共享客户端。
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    def _fire_stream_delta(self, text: str) -> None:
        """触发所有已注册的流式增量回调（显示 + TTS）。"""
        # 如果工具迭代设置了换行标志，在第一个真实文本增量前添加一个段落分隔符。
        # 这可以防止原始问题（跨工具边界的文本连接），同时避免在多个工具迭代连续运行时堆叠空白行。
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
        for cb in (self.stream_delta_callback, self._stream_callback):
            if cb is not None:
                try:
                    cb(text)
                except Exception:
                    pass

    def _fire_reasoning_delta(self, text: str) -> None:
        """如果已注册，则触发推理回调。"""
        self._reasoning_deltas_fired = True
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """通知显示层模型正在生成工具调用参数。

        当流式响应开始产生 tool_call / tool_use 令牌时，每个工具名称触发一次。
        这让 TUI 有机会显示加载动画或状态行，以免用户在生成大型工具负载（例如 45 KB 的 write_file）时盯着冻结的屏幕。
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """如果注册了任何流式消费者，则返回 True。"""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """用于实时令牌传递的 _interruptible_api_call 流式变体。

        处理所有三种 api_modes：
        - chat_completions: OpenAI 兼容端点上的 stream=True
        - anthropic_messages: 通过 Anthropic SDK 的 client.messages.stream()
        - codex_responses: 委托给 _run_codex_stream（已流式）

        为每个文本令牌触发 stream_delta_callback 和 _stream_callback。
        工具调用轮次抑制回调——只有纯文本最终响应流向消费者。
        返回模拟非流式响应形状的 SimpleNamespace，以便代理循环的其余部分不变。

        在指示不支持流式的 provider 错误时回退到 _interruptible_api_call。
        """
        if self.api_mode == "codex_responses":
            # Codex 通过 _run_codex_stream 在内部流式传输。_interruptible_api_call
            # 中的主调度已经调用它；我们只需要确保 on_first_delta 到达。
            # 暂时存储在实例上，以便 _run_codex_stream 可以拾取它。
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        result = {"response": None, "error": None}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # 跟踪是否有增量被触发（用于回退）
        # 最后一个真实流式块的墙上时钟时间戳。外层
        # 轮询循环使用它来检测持续接收
        # SSE keep-alive ping 但没有实际数据的过时连接。
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """Stream a chat completions response."""
            import httpx as _httpx
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 60.0))
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            # Reset stale-stream timer so the detector measures from this
            # attempt's start, not a previous attempt's last chunk.
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama 兼容端点在并行批次中为每个工具调用重用索引 0，
            # 仅通过 id 区分。按原始索引跟踪最后看到的 id，
            # 以便检测在同一索引开始的新工具调用并将其重定向到新槽。
            _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
            _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            # Reset per-call reasoning tracking so _build_assistant_message
            # knows whether reasoning was already displayed during streaming.
            self._reasoning_deltas_fired = False

            _first_chunk_seen = False
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                if not _first_chunk_seen:
                    _first_chunk_seen = True
                    self._touch_activity("receiving stream response")

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage comes in the final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # Accumulate reasoning content
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    else:
                        # Tool calls suppress regular content streaming (avoids
                        # displaying chatty "I'll use the tool..." text alongside
                        # tool calls).  But reasoning tags embedded in suppressed
                        # content should still reach the display — otherwise the
                        # reasoning box only appears as a post-response fallback,
                        # rendering it confusingly after the already-streamed
                        # response.  Route suppressed content through the stream
                        # delta callback so its tag extraction can fire the
                        # reasoning display.  Non-reasoning text is harmlessly
                        # suppressed by the CLI's _stream_delta when the stream
                        # box is already closed (tool boundary flush).
                        if self.stream_delta_callback:
                            try:
                                self.stream_delta_callback(delta.content)
                            except Exception:
                                pass

                # Accumulate tool call deltas — notify display on first name
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama 修复：检测重用相同
                        # 原始索引（不同 id）的新工具调用并重定向到新槽。
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # 当完整名称可用时，每个工具触发一次
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # Build mock response matching non-streaming shape
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    ))

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=finish_reason or "stop",
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """流式传输 Anthropic Messages API 响应。

            为实时令牌传递触发增量回调，但从 get_final_message() 返回原生 Anthropic 消息对象，
            以便代理循环的其余部分（验证、工具提取等）保持不变。
            """
            has_tool_use = False
            self._reasoning_deltas_fired = False

            # 为本次尝试重置过时流计时器
            last_chunk_time["t"] = time.time()
            # 使用 Anthropic SDK 的流式上下文管理器
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # 返回原生 Anthropic 消息以供后续处理
                return stream.get_final_message()

        def _call():
            import httpx as _httpx

            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                for _stream_attempt in range(_max_stream_retries + 1):
                    try:
                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()
                        return  # success
                    except Exception as e:
                        if deltas_were_sent["yes"]:
                            # 在交付一些令牌之后流式传输失败。不重试或回退 —— 部分内容已到达用户。
                            logger.warning(
                                "Streaming failed after partial delivery, not retrying: %s", e
                            )
                            result["error"] = e
                            return

                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # 来自代理的 SSE 错误事件（例如 OpenRouter 发送 {"error":{"message":"Network connection lost."}}）
                        # 会被 OpenAI SDK 抛出为 APIError。这些在语义上等同于 httpx 连接中断 —— 上游流已关闭 —— 
                        # 应该使用新连接重试。与 HTTP 错误区分开来：来自 SSE 的 APIError 没有 status_code，
                        # 而 APIStatusError (4xx/5xx) 始终有一个。
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_sse = str(e).lower()
                                _SSE_CONN_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # 瞬时网络 / 超时错误。首先使用新连接重试流式请求。
                            if _stream_attempt < _max_stream_retries:
                                logger.info(
                                    "Streaming attempt %s/%s failed (%s: %s), "
                                    "retrying with fresh connection...",
                                    _stream_attempt + 1,
                                    _max_stream_retries + 1,
                                    type(e).__name__,
                                    e,
                                )
                                self._emit_status(
                                    f"⚠️ Connection to provider dropped "
                                    f"({type(e).__name__}). Reconnecting… "
                                    f"(attempt {_stream_attempt + 2}/{_max_stream_retries + 1})"
                                )
                                # 在重试前关闭过时的请求客户端
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # 同时重建主客户端以清除连接池中的任何死连接。
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                continue
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                            logger.warning(
                                "Streaming exhausted %s retries on transient error, "
                                "falling back to non-streaming: %s",
                                _max_stream_retries + 1,
                                e,
                            )
                        else:
                            _err_lower = str(e).lower()
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Falling back to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery, falling back to non-streaming: %s",
                                e,
                            )

                        try:
                            # 重置过时计时器 —— 非流式回退使用其自身的客户端；防止过时探测器在来自失败流的过时时间戳上触发。
                            last_chunk_time["t"] = time.time()
                            result["response"] = self._interruptible_api_call(api_kwargs)
                        except Exception as fallback_err:
                            result["error"] = fallback_err
                        return
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # 为大上下文扩展过时超时：当上下文很大时，慢速模型（如 Opus）可能会在产生第一个令牌之前合理地思考几分钟。
        # 如果没有这一点，过时探测器会在模型的思考阶段关闭健康的连接，从而产生虚假的 RemoteProtocolError（“对端关闭连接”）。
        _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
        if _est_tokens > 100_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
        elif _est_tokens > 50_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
        else:
            _stream_stale_timeout = _stream_stale_timeout_base

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)

            # 探测过时流：连接由 SSE ping 保持活动，但没有交付真实块。关闭客户端，以便内部重试循环可以启动新连接。
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # Rebuild the primary client too — its connection pool
                # may hold dead sockets from the same provider outage.
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # Reset the timer so we don't kill repeatedly while
                # the inner thread processes the closure.
                last_chunk_time["t"] = time.time()

            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # Streaming failed AFTER some tokens were already delivered to
                # the platform.  Re-raising would let the outer retry loop make
                # a new API call, creating a duplicate message.  Return a
                # partial "stop" response instead so the outer loop treats this
                # turn as complete (no retry, no fallback).
                logger.warning(
                    "Partial stream delivered before error; returning stub "
                    "response to prevent duplicate messages: %s",
                    result["error"],
                )
                _stub_msg = SimpleNamespace(
                    role="assistant", content=None, tool_calls=None,
                    reasoning_content=None,
                )
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """切换到链中的下一个回退模型 / 提供商。

        当当前模型在重试后仍然失败时调用。就地交换 OpenAI 客户端、模型标识和提供商，以便重试循环可以继续使用新的后端。
        每次调用时都会在链中前进；当耗尽时返回 False。

        使用集中式提供商路由器 (resolve_provider_client) 进行身份验证解析和客户端构建 —— 没有重复的提供商 → 密钥映射。
        """
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback()  # skip invalid, try next

        # 使用集中式路由器进行客户端构建。
        # 由于主代理需要为 Codex 提供商直接访问 responses.stream()，因此 raw_codex=True。
        try:
            from agent.auxiliary_client import resolve_provider_client
            # 从回退配置中传递 base_url 和 api_key，以便自定义端点（例如 Ollama Cloud）能够正确解析，
            # 而不是回退到 OpenRouter 默认值。
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
            # when no explicit key is in the fallback config.
            if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _ = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True,
                explicit_base_url=fb_base_url_hint,
                explicit_api_key=fb_api_key_hint)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return self._try_activate_fallback()  # try next in chain

            # Determine api_mode from provider / base URL
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # Build native Anthropic client instead of using OpenAI client
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
                effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = getattr(fb_client, "base_url", None)
                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            else:
                # Swap OpenAI client and config in-place
                self.api_key = fb_client.api_key
                self.client = fb_client
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                }

            # Re-evaluate prompt caching for the new provider/model
            is_native_anthropic = fb_api_mode == "anthropic_messages"
            self._use_prompt_caching = (
                ("openrouter" in fb_base_url.lower() and "claude" in fb_model.lower())
                or is_native_anthropic
            )

            # 更新回退模型的上下文压缩器限制。
            # 如果没有这一点，压缩决策将使用主模型的上下文窗口（例如 200K）而不是回退模型的上下文窗口（例如 32K），
            # 导致超大规模会话溢出回退模型。
            if hasattr(self, 'context_compressor') and self.context_compressor:
                from agent.model_metadata import get_model_context_length
                fb_context_length = get_model_context_length(
                    self.model, base_url=self.base_url,
                    api_key=self.api_key, provider=self.provider,
                )
                self.context_compressor.model = self.model
                self.context_compressor.base_url = self.base_url
                self.context_compressor.api_key = self.api_key
                self.context_compressor.provider = self.provider
                self.context_compressor.context_length = fb_context_length
                self.context_compressor.threshold_tokens = int(
                    fb_context_length * self.context_compressor.threshold_percent
                )

            self._emit_status(
                f"🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback()  # try next in chain

    # ── Per-turn primary restoration ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """Restore the primary runtime at the start of a new turn.

        In long-lived CLI sessions a single AIAgent instance spans multiple
        turns.  Without restoration, one transient failure pins the session
        to the fallback provider for every subsequent turn.  Calling this at
        the top of ``run_conversation()`` makes fallback turn-scoped.

        The gateway creates a fresh agent per message so this is a no-op
        there (``_fallback_activated`` is always False at turn start).
        """
        if not self._fallback_activated:
            return False

        rt = self._primary_runtime
        try:
            # ── Core runtime state ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]           # setter updates _base_url_lower
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]

            # ── Rebuild client for the primary provider ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # 重置上下文压缩器状态
            cc = self.context_compressor
            cc.model = rt["compressor_model"]
            cc.base_url = rt["compressor_base_url"]
            cc.api_key = rt["compressor_api_key"]
            cc.provider = rt["compressor_provider"]
            cc.context_length = rt["compressor_context_length"]
            cc.threshold_tokens = rt["compressor_threshold_tokens"]

            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model, self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    # 定义指示瞬时传输故障的错误类型，这些错误值得使用重建的客户端/连接池再次尝试。
    _TRANSIENT_TRANSPORT_ERRORS = frozenset({
        "ReadTimeout", "ConnectTimeout", "PoolTimeout",
        "ConnectError", "RemoteProtocolError",
    })

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """针对瞬时传输故障尝试一次额外的 Primary Provider 恢复周期。

        在 max_retries 耗尽后，重建 Primary 客户端（清理陈旧连接池）并在回落（Falling back）前尝试最后一次。
        这对于直接端点（如 Anthropic, OpenAI, 本地模型）最有效，因为 TCP 层级的波动并不意味着 Provider 已宕机。

        对于已在服务端管理连接池和重试的聚合器 Provider（如 OpenRouter），此步骤会被跳过。
        """
        if self._fallback_activated:
            return False

        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in ("nous", "nous-research"):
            return False

        try:
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client, reason="primary_recovery", shared=True,
                    )
                except Exception:
                    pass

            rt = self._primary_runtime
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]

            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="primary_recovery",
                    shared=True,
                )

            wait_time = min(3 + retry_count, 8)
            self._vprint(
                f"{self.log_prefix}🔁 Transient {error_type} on {self.provider} — "
                f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
                force=True,
            )
            time.sleep(wait_time)
            return True
        except Exception as e:
            logging.warning("Primary transport recovery failed: %s", e)
            return False

    # ── End provider fallback ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        OpenCode Go keeps dots (e.g. minimax-m2.7)."""
        if (getattr(self, "provider", "") or "").lower() in {"alibaba", "opencode-go"}:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return "dashscope" in base or "aliyuncs" in base or "opencode.ai/zen/go" in base

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            # 传递 context_length，以便适配器在用户配置的上下文窗口小于模型输出限制时，能够限制 max_tokens（涉及上下文压缩策略）。
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
            )

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in self.base_url.lower()
                or "api.githubcopilot.com" in self.base_url.lower()
            )

            # 解析推理力度：配置优先于默认值 (medium)
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = self.session_id

            if reasoning_enabled:
                if is_github_responses:
                    # Copilot 的 Responses 路由声明支持 reasoning-effort，
                    # 但不支持 OpenAI 特有的 prompt cache 或加密推理字段。
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses:
                kwargs["include"] = []

            if self.max_tokens is not None:
                kwargs["max_output_tokens"] = self.max_tokens

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue

                # Codex 专有的重放状态不得泄露给严格的 chat-completions API。
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        # GPT-5 和 Codex 模型对 'developer' 角色的指令遵循效果优于 'system'。
        # 在 API 边界处转换角色，以便内部消息表示保持统一 ("system")。
        _model_lower = (self.model or "").lower()
        if (
            sanitized_messages
            and sanitized_messages[0].get("role") == "system"
            and any(p in _model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            sanitized_messages = list(sanitized_messages)
            sanitized_messages[0] = {**sanitized_messages[0], "role": "developer"}

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 1800.0)),
        }
        if self.tools:
            api_kwargs["tools"] = self.tools

        if self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))
        elif self._is_openrouter_url() and "claude" in (self.model or "").lower():
            # OpenRouter 会将请求转换为 Anthropic 的 Messages API，后者要求 max_tokens 为必填项。
            # 如果省略，OpenRouter 默认值可能过低，导致模型将预算消耗在思考上而几乎没有剩余空间给实际响应（尤其是大型工具调用）。
            # 发送模型的实际输出限制以确保完整容量。
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _model_output_limit = _get_anthropic_max_output(self.model)
                api_kwargs["max_tokens"] = _model_output_limit
            except Exception:
                pass

        extra_body = {}

        _is_openrouter = self._is_openrouter_url()
        _is_github_models = (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        )

        # Provider 偏好设置是 OpenRouter 特有的。仅发送到兼容 OpenRouter 的端点。
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self._base_url_lower

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if self.reasoning_config is not None:
                    rc = dict(self.reasoning_config)
                    # Nous Portal 要求启用推理 — 当禁用时不要发送 enabled=false（会导致 400）。
                    if _is_nous and rc.get("enabled") is False:
                        pass
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }

        # Nous Portal 产品归因
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        # Ollama num_ctx: 覆盖默认的 2048，使模型能够使用其训练时的完整上下文窗口（涉及上下文压缩）。
        if self._ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = self._ollama_num_ctx
            extra_body["options"] = options

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # xAI prompt 缓存：发送 x-grok-conv-id 标头以将请求路由到同一服务器，最大化自动缓存命中率（涉及 Prompt 冻结策略）。
        if "x.ai" in self._base_url_lower and hasattr(self, "session_id") and self.session_id:
            api_kwargs["extra_headers"] = {"x-grok-conv-id": self.session_id}

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if "nousresearch" in self._base_url_lower:
            return True
        if "ai-gateway.vercel.sh" in self._base_url_lower:
            return True
        if "models.github.ai" in self._base_url_lower or "api.githubcopilot.com" in self._base_url_lower:
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # 备选方案：当不存在结构化推理字段时，从内容中提取内联的 <think> 块
        # （某些模型/Provider 将思考过程直接嵌入内容中，而不是返回单独的 API 字段）。
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # 当流式传输激活时跳过回调 —— 推理过程已经通过以下路径显示：
            #   (a) _fire_reasoning_delta (结构化的 reasoning_content deltas)
            #   (b) _stream_delta 标签提取 (<think>/<REASONING_SCRATCHPAD>)
            # 当流式传输未激活时始终触发，以便非流式模式（网关、批处理、静默模式）仍能获取推理信息。
            if not self.stream_delta_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # 将 reasoning_details 原样传回，以便 Provider (OpenRouter, Anthropic, OpenAI) 跨回合保持推理连贯性。
            # 每个 Provider 可能包含必须精确保留的不透明字段（签名、加密内容）。
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API：保留加密的推理项以实现多轮连贯性。这些项在下一轮中作为输入重放。
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # 保留 extra_content（如 Gemini 的 thought_signature），以便在后续 API 调用中发回。
                # 如果没有这个，Gemini 3 推理模型会拒绝请求并返回 400 错误。
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """为严格的 API 剥离 tool_calls 中的 Codex Responses 字段。

        像 Mistral、Fireworks 以及其他严格兼容 OpenAI 的 API 会验证聊天补全模式，
        并拒绝未知字段（call_id, response_item_id）。这些字段保留在内部消息历史中 —— 本方法仅修改传出的 API 副本。
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    def _should_sanitize_tool_calls(self) -> bool:
        """确定 tool_calls 是否需要针对严格 API 进行清理。"""
        return self.api_mode != "codex_responses"

    def flush_memories(self, messages: list = None, min_turns: int = None):
        """在上下文丢失前，给模型一次 Turn 的机会来持久化记忆。

        在上下文压缩、会话重置或 CLI 退出前调用。注入一条 flush 消息，进行一次 API 调用，
        执行任何记忆工具调用，然后从消息列表中清除所有 flush 痕迹。
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                api_msg.pop("_thinking_prefill", None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()
                return

            # 当辅助客户端可用时使用它 —— 它更便宜且能避免 Codex Responses API 的不兼容问题。
            from agent.auxiliary_client import call_llm as _call_llm
            _aux_available = True
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=0.3,
                    max_tokens=5120,
                    timeout=30.0,
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = 0.3
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_kwargs as _build_ant_kwargs
                ant_kwargs = _build_ant_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": 0.3,
                    **self._max_tokens_param(5120),
                }
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(**api_kwargs, timeout=30.0)

            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import normalize_anthropic_response as _nar_flush
                _flush_msg, _ = _nar_flush(response, strip_tool_prefix=self._is_anthropic_oauth)
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif hasattr(response, "choices") and response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # 清除 flush 痕迹：从 flush 消息开始移除所有内容。
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    # ================================================================
    # 【文档锚点 4A】压缩与 session lineage
    # 1. 压缩前先执行 flush_memories()，给模型一次只带 memory 工具的补录机会（见 run_agent.py:5677-5834）
    # 2. _compress_context() 负责 session 分裂和新 session 建立（见 run_agent.py:5835-5928）
    # 3. 真正的摘要算法在 ContextCompressor.compress()（见 agent/context_compressor.py:565-696）
    # ================================================================
    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default") -> tuple:
        """压缩会话上下文并分裂 SQLite 中的 session"""
        # 【文档锚点 4A】压缩前先 flush memory，再生成压缩摘要并创建新的 lineage session
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
        )
        # 压缩前先执行 flush_memories()，给模型一次只带 memory 工具的补录机会
        self.flush_memories(messages, min_turns=0)

        # 压缩前通知外部记忆管理器，以免丢弃上下文
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # 将标题传播到具有自动编号的新会话中
                old_title = self._session_db.get_session_title(self.session_id)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # 重置 Flush 游标 —— 新会话从零开始写入消息
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # 压缩后更新 Token 估算，以便压力计算使用压缩后的计数（涉及上下文压缩策略）。
        _compressed_est = (
            estimate_tokens_rough(new_system_prompt)
            + estimate_messages_tokens_rough(compressed)
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # 仅当压缩确实使我们低于警告级别时，才重置压力警告（涉及上下文压缩策略）。
        if self.context_compressor.threshold_tokens > 0:
            _post_progress = _compressed_est / self.context_compressor.threshold_tokens
            if _post_progress < 0.85:
                self._context_pressure_warned = False

        # 清除文件读取去重缓存。压缩后原始内容已被摘要化，如果模型重新读取相同文件，它需要完整内容。
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt

    # ================================================================
    # 【文档锚点 3D】工具执行
    # 1. run_conversation() 在接收到 assistant.tool_calls 后先做参数清洗和去重，再执行工具
    # 2. _execute_tool_calls() 决定走并发还是顺序分支（见 run_agent.py:5930-5951）
    # 3. 并发分支在 _execute_tool_calls_concurrent()（见 run_agent.py:6027-6249）
    # 4. 顺序分支在 _execute_tool_calls_sequential()，其中 todo/memory/session_search/delegate_task
    #    会被 agent loop 特判（见 run_agent.py:6251-6360）
    # 5. 其他工具统一回落到 handle_function_call()，再由 registry dispatch 到真实 handler
    #    （见 model_tools.py:459-548）
    # ================================================================
    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """执行工具调用（涉及工具分层分发策略）"""
        # 【文档锚点 3D】tool_calls 进入 agent loop 后，会先在这里决定并发/串行分支
        tool_calls = assistant_message.tool_calls

        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None) -> str:
        """调用单个工具并返回结果字符串（涉及工具分层分发策略）。"""
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
            )

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag)
        results = [None] * num_tools

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id, tool_call.id)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error)

        # Start spinner for CLI mode (skip when TUI handles tool progress)
        spinner = None
        if self.quiet_mode and not self.tool_progress_callback and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # Wait for all to complete (exceptions are captured inside _run_tool)
                concurrent.futures.wait(futures)
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # Shouldn't happen, but safety fallback
                function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self.quiet_mode:
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            )

            # 从工具参数中发现子目录上下文文件（涉及工具分层分发策略）
            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── 每回合累计预算强制执行（涉及 Turn 路由与预算治理） ──
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

        # ── 预算压力注入（涉及 Turn 路由中的迭代限制） ──
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️  WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """顺序执行工具调用（涉及工具分层分发策略）。用于单次调用或交互式工具。"""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # 安全检查：在启动每个工具前检查中断。如果用户请求中断，跳过所有剩余工具。
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            # Reset nudge counters when the relevant tool is actually used
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            self._current_tool = function_name
            self._touch_activity(f"executing tool: {function_name}")

            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                from tools.delegate_tool import delegate_task as _delegate_task
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self.quiet_mode and not self.tool_progress_callback and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # 记忆提供者工具 (hindsight_retain, honcho_search 等)。
                # 这些工具不在工具注册中心 —— 通过 MemoryManager 进行路由（涉及工具分层分发策略）。
                spinner = None
                if self.quiet_mode and not self.tool_progress_callback:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if not self.tool_progress_callback:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    else:
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, len(function_result))

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            )

            # 从工具参数中发现子目录上下文文件（涉及工具分层分发策略）
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── 每回合累计预算强制执行（涉及 Turn 路由与预算治理） ──
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))

        # ── 预算压力注入（涉及 Turn 路由中的迭代限制） ──
        # 在处理完本轮所有工具调用后，检查是否接近 max_iterations。
        # 如果是，则在最后一个工具结果的 JSON 中注入警告，以便 LLM 在读取结果时能自然看到。
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️  WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

    def _get_budget_warning(self, api_call_count: int) -> Optional[str]:
        """返回预算压力字符串，如果尚未需要则返回 None（涉及 Turn 路由预算管理）。

        采用两级提示系统：
          - 警示 (70%)：提示模型整合当前工作
          - 警告 (90%)：紧急提示，要求模型立即给出最终答复
        """
        if not self._budget_pressure_enabled or self.max_iterations <= 0:
            return None
        progress = api_call_count / self.max_iterations
        remaining = self.max_iterations - api_call_count
        if progress >= self._budget_warning_threshold:
            return (
                f"[BUDGET WARNING: Iteration {api_call_count}/{self.max_iterations}. "
                f"Only {remaining} iteration(s) left. "
                "Provide your final response NOW. No more tool calls unless absolutely critical.]"
            )
        if progress >= self._budget_caution_threshold:
            return (
                f"[BUDGET: Iteration {api_call_count}/{self.max_iterations}. "
                f"{remaining} iterations left. Start consolidating your work.]"
            )
        return None

    def _emit_context_pressure(self, compaction_progress: float, compressor) -> None:
        """Notify the user that context is approaching the compaction threshold.

        Args:
            compaction_progress: How close to compaction (0.0–1.0, where 1.0 = fires).
            compressor: The ContextCompressor instance (for threshold/context info).

        Purely user-facing — does NOT modify the message stream.
        For CLI: prints a formatted line with a progress bar.
        For gateway: fires status_callback so the platform can send a chat message.
        """
        from agent.display import format_context_pressure, format_context_pressure_gateway

        threshold_pct = compressor.threshold_tokens / compressor.context_length if compressor.context_length else 0.5

        # CLI output — always shown (these are user-facing status notifications,
        # not verbose debug output, so they bypass quiet_mode).
        # Gateway users also get the callback below.
        if self.platform in (None, "cli"):
            line = format_context_pressure(
                compaction_progress=compaction_progress,
                threshold_tokens=compressor.threshold_tokens,
                threshold_percent=threshold_pct,
                compression_enabled=self.compression_enabled,
            )
            self._safe_print(line)

        # Gateway / external consumers
        if self.status_callback:
            try:
                msg = format_context_pressure_gateway(
                    compaction_progress=compaction_progress,
                    threshold_percent=threshold_pct,
                    compression_enabled=self.compression_enabled,
                )
                self.status_callback("context_pressure", msg)
            except Exception:
                logger.debug("status_callback error in context pressure", exc_info=True)

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """当达到最大迭代次数时请求总结。返回最终响应文本。"""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # 清理严格 API（如 Mistral）不支持的内部字段，防止 422 错误
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (assistant_message.content or "").strip() if assistant_message else ""
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # 注入 Provider 路由偏好
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak, normalize_anthropic_response as _nar
                    _ant_kw = _bak(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # 重试总结生成
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (retry_msg.content or "").strip() if retry_msg else ""
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak2, normalize_anthropic_response as _nar2
                    _ant_kw2 = _bak2(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        单次请求生命周期

        run_conversation() 不是"一次 API 调用"，而是一段多轮编排。

        进入主循环前：
        1. 恢复 primary runtime，清理 fallback 状态（见 run_agent.py:6832-6863）
        2. 复制历史消息，剥离预算警告，必要时回填 todo store（见 run_agent.py:6892-6907）
        3. 追加当前 user message，并记录持久化索引（见 run_agent.py:6917-6937）
        4. 如历史已经逼近 context window，则先做 preflight compression（见 run_agent.py:6993-7049）

        主循环内部：
        1. 主循环入口在 while api_call_count < self.max_iterations（见 run_agent.py:7111-7128）
        2. Provider 适配集中在 _build_api_kwargs()（见 run_agent.py:5229-5450）
        3. 推理相关 extra_body 是否支持，由 _supports_reasoning_extra_body() 判定（见 run_agent.py:5452-5484）
        4. 请求前还会统一清理内部字段、注入 reasoning content、修复 API message 结构（见 run_agent.py:7190-7250）
        """
        _install_safe_stdio()

        # ================================================================
        # 【进入主循环前 - 1】恢复 primary runtime，清理 fallback 状态
        # 如果上一轮激活了 fallback，则恢复首选模型，使本轮获得新的尝试机会
        # ================================================================
        self._restore_primary_runtime()

        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        effective_task_id = task_id or str(uuid.uuid4())

        # 重试计数器和迭代预算在每轮开始时重置，防止上一轮的子 agent 用量影响本轮
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._thinking_prefill_retries = 0
        self._last_content_with_tools = None
        self._mute_post_response = False
        self._surrogate_sanitized = False

        # 预轮连接健康检查：清理上一轮 provider 故障遗留的死 TCP 连接
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 Detected stale connections from a previous provider "
                        "issue — cleaned up automatically. Proceeding with fresh "
                        "connection."
                    )
            except Exception:
                pass

        self.iteration_budget = IterationBudget(self.max_iterations)

        _msg_preview = (user_message[:80] + "...") if len(user_message) > 80 else user_message
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # ================================================================
        # 【进入主循环前 - 2】复制历史消息，剥离预算警告，回填 todo store
        # ================================================================
        messages = list(conversation_history) if conversation_history else []

        # 剥离历史消息中的预算警告（GPT 模型会将其误解为持续性指令）
        if messages:
            _strip_budget_warnings_from_history(messages)

        # Gateway 每条消息创建一个新 AIAgent，需要从历史中恢复 todo 状态
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)

        # prefill_messages 在 API 调用时注入，不保存到消息列表（ephemeral）
        self._user_turn_count += 1
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # 跟踪记忆提醒触发（基于 Turn，在此检查）。
        # Skill 触发在 Agent 循环完成后检查，基于本轮使用的工具迭代次数。
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # 添加用户消息
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            self._safe_print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")

        # ================================================================
        # 【文档锚点 3A】System Prompt 冻结与易变上下文注入
        # run_conversation() 会优先复用 session 中已经保存的 system prompt；
        # 只有首次进入会话或压缩后才重建（见 run_agent.py:6952-6991）
        # 真正的 prompt 构建逻辑在 _build_system_prompt()（见 run_agent.py:2599-2610）
        # ================================================================

        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass

            if stored_prompt:
                # 继续会话：从 session DB 加载已有的 system prompt，保持 Anthropic 前缀缓存一致
                self._cached_system_prompt = stored_prompt
            else:
                # 新会话首次：构建 system prompt
                self._cached_system_prompt = self._build_system_prompt(system_message)
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt

        # ================================================================
        # 【文档锚点 4A】进入主循环前 - 前置上下文压缩 (Preflight compression)
        # 在进入主循环前检查加载的历史是否已超过模型的 context threshold
        # 处理用户切换到较小 context window 模型的情况
        # ================================================================
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
                    )
                # 对于具有较小上下文窗口的大型会话，可能需要多次压缩（每次总结中间的 N 轮）。
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # 无法进一步压缩
                    # 压缩创建了新会话 — 清理历史引用，以便 _flush_messages_to_session_db 将所有
                    # 压缩后的消息写入新会话的 SQLite，而不是因为 conversation_history 仍是
                    # 压缩前的长度而跳过它们。
                    conversation_history = None
                    # 压缩后重新估算
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # 低于阈值

        # 插件钩子：pre_llm_call
        # 在工具调用循环之前的每一轮触发一次。插件可以返回一个带有 ``context`` 键的字典（或纯字符串），
        # 其值会被追加到当前轮次的用户消息中。
        #
        # 上下文始终被注入到用户消息中，而非系统提示词。这样可以保留提示词缓存前缀——
        # 系统提示词在各轮次间保持相同，因此可以复用缓存的 token。系统提示词是 Hermes 的领地；
        # 插件只在用户输入旁边贡献上下文。
        #
        # 所有注入的上下文都是瞬时的（不会持久化到 session DB）。
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        # 主对话循环
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        
        # 清除所有过时的中断状态
        self.clear_interrupt()

        # ================================================================
        # 【文档锚点 3B】易变内容注入：外部记忆 provider 预取
        # 外部记忆 provider 先 prefetch_all() 一次，结果缓存到 turn 内变量
        # 记忆块被 build_memory_context_block() 包进 <memory-context>，
        # 并注入到当前 turn 的 user message（见 agent/memory_manager.py:54-69）
        # ================================================================
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""  # 【文档锚点 3B】本轮记忆预取结果只存在于局部变量
            except Exception:
                pass

        # ================================================================
        # 【文档锚点 3C】主循环内部 - 主循环入口
        # while api_call_count < self.max_iterations ...
        # ================================================================
        while api_call_count < self.max_iterations and self.iteration_budget.remaining > 0:  # 【文档锚点 3C】单个 turn 的核心编排循环
            # 重置每轮检查点去重，以便每次迭代都能获取一个快照
            self._checkpoint_mgr.new_turn()

            # 检查中断请求（例如用户发送了新消息）
            if self._interrupt_requested:
                interrupted = True
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")
            if not self.iteration_budget.consume():
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break

            # 触发 step_callback 以便 gateway hooks (agent:step 事件)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # 跟踪工具调用迭代次数以进行 skill nudge。
            # 当 skill_manage 实际被使用时，计数器会重置。
            if (self._skill_nudge_interval > 0
                    and "skill_manage" in self.valid_tool_names):
                self._iters_since_skill += 1
            
            # 准备 API 调用的消息
            # 如果有临时系统提示词，将其前置到消息列表
            # 注意：推理内容通过 <think> 标签嵌入到 content 中以进行轨迹存储。
            # 然而，像 Moonshot AI 这样的提供商要求在带有 tool_calls 的 assistant 消息上
            # 使用单独的 'reasoning_content' 字段。我们在这里处理这两种情况。
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                # 将临时上下文注入当前轮次的用户消息中。
                # 来源：memory manager 预取 + plugin pre_llm_call hooks
                # 目标为 "user_message"（默认）。两者都是
                # API 调用时注入——`messages` 中的原始消息从不被修改，
                # 因此不会泄漏到 session 持久化中。
                if idx == current_turn_user_idx and msg.get("role") == "user":
                    _injections = []
                    if _ext_prefetch_cache:
                        _fenced = build_memory_context_block(_ext_prefetch_cache)  # 【文档锚点 3B】把 recall 结果包进 <memory-context> 再注入当前 user message
                        if _fenced:
                            _injections.append(_fenced)
                    if _plugin_user_context:
                        _injections.append(_plugin_user_context)
                    if _injections:
                        _base = api_msg.get("content", "")
                        if isinstance(_base, str):
                            api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

                # 对于所有 assistant 消息，将 reasoning 传回 API
                # 这确保了多轮推理上下文的保留
                if msg.get("role") == "assistant":
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # 添加 reasoning_content 以提高 API 兼容性（Moonshot AI、Novita、OpenRouter）
                        api_msg["reasoning_content"] = reasoning_text

                # 移除 'reasoning' 字段——它仅用于轨迹存储
                # 我们已经将其复制到 'reasoning_content' 供 API 使用
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # 移除 finish_reason——严格的 API（如 Mistral）不接受此字段
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # Strip internal thinking-prefill marker
                api_msg.pop("_thinking_prefill", None)
                # Strip Codex Responses API fields (call_id, response_item_id) for
                # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
                # Uses new dicts so the internal messages list retains the fields
                # for Codex Responses compatibility.
                if self._should_sanitize_tool_calls():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # 保留 'reasoning_details'——OpenRouter 使用它来维护多轮推理上下文
                # 签名字段有助于保持推理的连续性
                api_messages.append(api_msg)

            # ================================================================
            # 【文档锚点 3B】易变内容注入 - 续：ephemeral_system_prompt 和 prefill_messages
            # ephemeral_system_prompt 只在 API 发送前和 cached system 拼接，不进入持久化 transcript
            # prefill_messages 也只在 API payload 中插入，不写回 session
            # ================================================================
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            # Plugin context 从 pre_llm_call hooks 注入到 user message（见上面 injection block）
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # prefill_messages 在 system prompt 之后、conversation history 之前插入
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # 通过 OpenRouter 为 Claude 模型应用 Anthropic 提示缓存。
            # 自动检测：如果模型名包含 "claude" 且 base_url 是 OpenRouter，
            # 则注入 cache_control 断点（system + 最近 3 条消息）以将多轮对话的
            # 输入 token 成本降低约 75%。
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=self._cache_ttl, native_anthropic=(self.api_mode == 'anthropic_messages'))

            # 安全网：在发送到 API 之前，剥离孤立的工具结果/为缺失的结果添加桩。
            # 无条件运行——不受 context_compressor 控制——因此来自 session 加载或
            # 手动消息操作的孤立项始终会被捕获。
            api_messages = self._sanitize_api_messages(api_messages)

            # 计算近似请求大小以供日志记录
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4  # 粗略估算：每个 token 约 4 个字符
            
            # 安静模式下的思考旋转动画（API 调用期间）
            thinking_spinner = None

            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # 安静模式下的动画思考旋转器
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
                if self.thinking_callback:
                    # CLI TUI 模式：使用 prompt_toolkit 小部件而非原始旋转器（在流式和非流式模式下均可工作）
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # 仅在无流消费者且旋转器输出有安全接收端时使用原始 KawaiiSpinner
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()

            # 记录详细请求日志
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted=False
            anthropic_auth_retry_attempted=False
            nous_auth_retry_attempted=False
            thinking_sig_retry_attempted = False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # 防止在所有重试失败时出现 UnboundLocalError

            while retry_count < max_retries:
                try:
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                    try:
                        from hermes_cli.plugins import invoke_hook as _invoke_hook
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            session_id=self.session_id or "",
                            platform=self.platform or "",
                            model=self.model,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_mode=self.api_mode,
                            api_call_count=api_call_count,
                            message_count=len(api_messages),
                            tool_count=len(self.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=self.max_tokens,
                        )
                    except Exception:
                        pass

                    if env_var_enabled("HERMES_DUMP_REQUESTS"):
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # 始终优先选择流式路径——即使没有流消费者。
                    # 流式提供了精细的健康检查（90秒陈旧流检测，60秒读取超时），
                    # 而非流式路径缺乏这些。如果没有这个，子 Agent 和其他安静模式调用者
                    # 在 Provider 保持连接处于 SSE ping 状态但从不交付响应时可能会无限期挂起。
                    # 当没有注册消费者时，流式路径对回调是无操作的，并且如果 Provider 不支持，
                    # 会自动回退到非流式。
                    def _stop_spinner():
                        nonlocal thinking_spinner
                        if thinking_spinner:
                            thinking_spinner.stop("")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                    _use_streaming = True
                    if not self._has_stream_consumers():
                        # 无显示/TTS 消费者。仍然优先使用流式进行健康检查，
                        # 但跳过测试中的 Mock 客户端（mock 返回 SimpleNamespace，而非流式迭代器）。
                        from unittest.mock import Mock
                        if isinstance(getattr(self, "client", None), Mock):
                            _use_streaming = False

                    if _use_streaming:
                        response = self._interruptible_streaming_api_call(
                            api_kwargs, on_first_delta=_stop_spinner
                        )
                    else:
                        response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # 静默停止思考旋转器——随后的响应框或工具执行消息更有信息量。
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # 如果可用，记录带有提供商信息的响应
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # 在继续之前验证响应格式
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        output_items = getattr(response, "output", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(output_items, list):
                            response_invalid = True
                            error_details.append("response.output is not a list")
                        elif not output_items:
                            # 流回填可能失败，但 _normalize_codex_response 仍可以从 response.output_text 恢复。
                            # 只有当该回退也不存在时才标记为无效。
                            _out_text = getattr(response, "output_text", None)
                            _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                            if _out_text_stripped:
                                logger.debug(
                                    "Codex response.output is empty but output_text is present "
                                    "(%d chars); deferring to normalization.",
                                    len(_out_text_stripped),
                                )
                            else:
                                _resp_status = getattr(response, "status", None)
                                _resp_incomplete = getattr(response, "incomplete_details", None)
                                logger.warning(
                                    "Codex response.output is empty after stream backfill "
                                    "(status=%s, incomplete_details=%s, model=%s). %s",
                                    _resp_status, _resp_incomplete,
                                    getattr(response, "model", None),
                                    f"api_mode={self.api_mode} provider={self.provider}",
                                )
                                response_invalid = True
                                error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        content_blocks = getattr(response, "content", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(content_blocks, list):
                            response_invalid = True
                            error_details.append("response.content is not a list")
                        elif not content_blocks:
                            response_invalid = True
                            error_details.append("response.content is empty")
                    else:
                        if response is None or not hasattr(response, 'choices') or response.choices is None or not response.choices:
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, 'choices'):
                                error_details.append("response has no 'choices' attribute")
                            elif response.choices is None:
                                error_details.append("response.choices is None")
                            else:
                                error_details.append("response.choices is empty")

                    if response_invalid:
                        # 在打印错误消息之前停止旋转器
                        if thinking_spinner:
                            thinking_spinner.stop("(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # 这通常是限流或提供商返回格式错误的响应
                        retry_count += 1

                        # 积极回退：空/格式错误的响应是常见的限流症状。立即切换到回退，而不是用扩展退避重试。
                        if self._fallback_index < len(self._fallback_chain):
                            self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue

                        # 检查响应中的错误字段（某些提供商包含此字段）
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # 尝试从错误元数据中提取提供商
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # 尝试从 model 字段获取提供商（OpenRouter 通常返回实际使用的模型）
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        if provider_name == "Unknown" and response:
                            # 记录所有响应属性以供调试
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                        
                        self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                        cleaned_provider_error = self._clean_error_message(error_msg)
                        self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  Response time: {api_duration:.2f}s (fast response often indicates rate limiting)", force=True)
                        
                        if retry_count >= max_retries:
                            # 放弃前尝试回退
                            self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                continue
                            self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Invalid API response shape. Likely rate limited or malformed provider response.",
                                "failed": True  # 标记为失败以便过滤
                            }
                        
                        # 限流的长退避（可能是 None choices 的原因）
                        # 抖动指数退避：5s 基础，120s 上限 + 随机抖动
                        wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                        self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time}s (extended backoff for possible rate limit)...", force=True)
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # 分段睡眠以保持对中断的响应能力
                        sleep_end = time.time() + wait_time
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted: retrying API call after rate limit (retry {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                        continue  # 重试 API 调用

                    # 在继续之前检查 finish_reason
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(response, "incomplete_details", None)
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(incomplete_details, "reason", None)
                        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                        finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                    else:
                        finish_reason = response.choices[0].finish_reason

                    if finish_reason == "length":
                        self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                        # ── 检测迭代预算（推理部分）耗尽 ──────────────
                        # 当模型将所有输出 token 都花在推理上而没有剩余用于响应时，继续重试毫无意义。
                        # 尽早检测到这一点并给出有针对性的错误，而不是浪费迭代预算。
                        _trunc_content = None
                        if self.api_mode == "chat_completions":
                            _trunc_msg = response.choices[0].message if (hasattr(response, "choices") and response.choices) else None
                            _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                        elif self.api_mode == "anthropic_messages":
                            # Anthropic response.content 是块列表
                            _text_parts = []
                            for _blk in getattr(response, "content", []):
                                if getattr(_blk, "type", None) == "text":
                                    _text_parts.append(getattr(_blk, "text", ""))
                            _trunc_content = "\n".join(_text_parts) if _text_parts else None

                        _thinking_exhausted = (
                            _trunc_content is not None
                            and not self._has_content_after_think_block(_trunc_content)
                        ) or _trunc_content is None

                        if _thinking_exhausted:
                            _exhaust_error = (
                                "Model used all output tokens on reasoning with none left "
                                "for the response. Try lowering reasoning effort or "
                                "increasing max_tokens."
                            )
                            self._vprint(
                                f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                                f"no visible response was produced.",
                                force=True,
                            )
                            # 返回用户友好的消息作为响应，以便 CLI 和 Gateway 都能自然显示。
                            _exhaust_response = (
                                "⚠️ **Thinking Budget Exhausted**\n\n"
                                "The model used all its output tokens on reasoning "
                                "and had none left for the actual response.\n\n"
                                "To fix this:\n"
                                "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                                "→ Increase the output token limit: "
                                "set `model.max_tokens` in config.yaml"
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": _exhaust_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _exhaust_error,
                            }

                        if self.api_mode == "chat_completions":
                            assistant_message = response.choices[0].message
                            if not assistant_message.tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += assistant_message.content

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: Your previous response was truncated by the output "
                                            "length limit. Continue exactly where you left off. Do not "
                                            "restart or repeat prior text. Finish the answer directly.]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response remained truncated after 3 continuation attempts",
                                }

                        # 如果有先前的消息，回滚到上一个完整的助手轮次
                        if len(messages) > 1:
                            self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit"
                            }
                        else:
                            # 第一条消息就被截断 - 标记为失败
                            self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit"
                            }
                    
                    # 跟踪响应中的实际 token 用量以进行上下文管理
                    if hasattr(response, 'usage') and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # 成功调用后缓存发现的上下文长度。
                        # 只持久化提供商确认的限制（从错误消息中解析的），而不是猜测的探测层级。
                        if self.context_compressor._context_probed:
                            ctx = self.context_compressor.context_length
                            if getattr(self.context_compressor, "_context_probe_persistable", False):
                                save_context_length(self.model, self.base_url, ctx)
                                self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False
                            self.context_compressor._context_probe_persistable = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        # 记录 API 调用详情以供调试/可观测性
                        _cache_pct = ""
                        if canonical_usage.cache_read_tokens and prompt_tokens:
                            _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                        logger.info(
                            "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                            self.session_api_calls, self.model, self.provider or "unknown",
                            prompt_tokens, completion_tokens, total_tokens,
                            api_duration, _cache_pct,
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # 将 token 计数持久化到 session DB 以供 /insights 使用。
                        # 为每个带有 session_id 的平台执行此操作，以便非 CLI
                        # 会话（gateway、cron、委托运行）不会在更高级别的持久化路径
                        # 被跳过或失败时丢失 token/账户数据。Gateway/session-store 写入使用
                        # 绝对总数，因此它们可以安全地覆盖这些每调用增量，
                        # 而不是重复计算。
                        if self._session_db and self.session_id:
                            try:
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=float(cost_result.amount_usd)
                                    if cost_result.amount_usd is not None else None,
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode="subscription_included"
                                    if cost_result.status == "included" else None,
                                    model=self.model,
                                )
                            except Exception:
                                pass  # 永远不要阻塞 Agent 循环
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # 启用提示缓存时记录缓存命中统计
                        if self._use_prompt_caching:
                            if self.api_mode == "anthropic_messages":
                                # Anthropic 使用 cache_read_input_tokens / cache_creation_input_tokens
                                cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                                written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                            else:
                                # OpenRouter 使用 prompt_tokens_details.cached_tokens
                                details = getattr(response.usage, 'prompt_tokens_details', None)
                                cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                                written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                self._vprint(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")
                    
                    has_retried_429 = False  # 成功后重置
                    self._touch_activity(f"API call #{api_call_count} completed")
                    break  # 成功，退出重试循环

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                    break

                except Exception as api_error:
                    # 在打印错误消息之前停止旋转器
                    if thinking_spinner:
                        thinking_spinner.stop("(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # ── 代理字符恢复 ───────────────────
                    # 当消息包含孤立的代理（U+D800..U+DFFF）时会发生 UnicodeEncodeError，
                    # 这些是无效的 UTF-8。通常来自富文本编辑器粘贴。
                    if isinstance(api_error, UnicodeEncodeError) and not getattr(self, '_surrogate_sanitized', False):
                        self._surrogate_sanitized = True
                        if _sanitize_messages_surrogates(messages):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                force=True,
                            )
                            continue

                    status_code = getattr(api_error, "status_code", None)
                    error_context = self._extract_api_error_context(api_error)
                    recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                        status_code=status_code,
                        has_retried_429=has_retried_429,
                        error_context=error_context,
                    )
                    if recovered_with_pool:
                        continue
                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, '_anthropic_api_key')
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token
                        if self._try_refresh_anthropic_client_credentials():
                            print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                            continue
                        # 凭据刷新没有帮助——显示诊断信息
                        key = self._anthropic_api_key
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
                        print(f"{self.log_prefix}   Auth method: {auth_method}")
                        print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                        print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                        print(f"{self.log_prefix}     • For API keys: verify at https://console.anthropic.com/settings/keys")
                        print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                        print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                    # ── 思考块签名恢复 ─────────────────
                    # Anthropic 根据完整轮次内容对思考块进行签名。任何上游变更（上下文压缩、截断、合并）都会使签名失效。
                    # 恢复方法：从所有消息中剥离 reasoning_details，以便下一次重试不发送任何思考块。
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 400
                        and not thinking_sig_retry_attempted
                    ):
                        _err_msg_lower = str(api_error).lower()
                        if "signature" in _err_msg_lower and "thinking" in _err_msg_lower:
                            thinking_sig_retry_attempted = True
                            for _m in messages:
                                if isinstance(_m, dict):
                                    _m.pop("reasoning_details", None)
                            self._vprint(
                                f"{self.log_prefix}⚠️  Thinking block signature invalid — "
                                f"stripped all thinking blocks, retrying...",
                                force=True,
                            )
                            logging.warning(
                                "%sThinking block signature recovery: stripped "
                                "reasoning_details from %d messages",
                                self.log_prefix, len(messages),
                            )
                            continue

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    _error_summary = self._summarize_api_error(api_error)
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        _error_summary,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                    self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
                    if status_code and status_code < 500:
                        _err_body = getattr(api_error, "body", None)
                        _err_body_str = str(_err_body)[:300] if _err_body else None
                        if _err_body_str:
                            self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                    # 在决定重试之前检查中断
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }

                    # 在通用 4xx 处理程序之前检查 413 payload-too-large。
                    # 413 是 payload 大小错误——正确的响应是执行上下文压缩并重试，而不是立即中止。
                    status_code = getattr(api_error, "status_code", None)

                    # ── Anthropic Sonnet 长上下文层级门控 ───────────
                    # 当 Claude Max（或类似）订阅不包括 1M 上下文层级时，Anthropic 返回 HTTP 429 "Extra usage is required for long context requests"。
                    # 这不是瞬时限流——重试或切换凭据无济于事。将上下文减少到 200k（标准层级）并压缩。
                    # 仅适用于 Sonnet——Opus 1M 是通用访问。
                    _is_long_context_tier_error = (
                        status_code == 429
                        and "extra usage" in error_msg
                        and "long context" in error_msg
                        and "sonnet" in self.model.lower()
                    )
                    if _is_long_context_tier_error:
                        _reduced_ctx = 200000
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length
                        if old_ctx > _reduced_ctx:
                            compressor.context_length = _reduced_ctx
                            compressor.threshold_tokens = int(
                                _reduced_ctx * compressor.threshold_percent
                            )
                            compressor._context_probed = True
                            # 不要持久化——这是订阅层级限制，而不是模型能力。
                            # 如果用户稍后启用了额外用量，1M 限制应该会自动恢复。
                            compressor._context_probe_persistable = False
                            self._vprint(
                                f"{self.log_prefix}⚠️  Anthropic long-context tier "
                                f"requires extra usage — reducing context: "
                                f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts <= max_compression_attempts:
                            original_len = len(messages)
                            messages, active_system_prompt = self._compress_context(
                                messages, system_message,
                                approx_tokens=approx_tokens,
                                task_id=effective_task_id,
                            )
                            if len(messages) < original_len or old_ctx > _reduced_ctx:
                                self._emit_status(
                                    f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                    f"(was {old_ctx:,}), retrying..."
                                )
                                time.sleep(2)
                                restart_with_compressed_messages = True
                                break

                    # 限流错误（429 或配额耗尽）的积极回退。
                    # 当配置了回退模型时，立即切换而不是用指数退避消耗重试——主要提供商在重试窗口内不会恢复。
                    is_rate_limited = (
                        status_code == 429
                        or "rate limit" in error_msg
                        or "too many requests" in error_msg
                        or "rate_limit" in error_msg
                        or "usage limit" in error_msg
                        or "quota" in error_msg
                    )
                    if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                        # 如果凭据池轮换可能仍然恢复，则不要积极回退。
                        pool = self._credential_pool
                        pool_may_recover = pool is not None and pool.has_available()
                        if not pool_may_recover:
                            self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                continue

                    is_payload_too_large = (
                        status_code == 413
                        or 'request entity too large' in error_msg
                        or 'payload too large' in error_msg
                        or 'error code: 413' in error_msg
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True
                            }
                        self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # 压缩重试之间的短暂暂停
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True
                            }

                    # 在通用 4xx 处理程序之前检查上下文长度错误。
                    # 本地后端（LM Studio、Ollama、llama.cpp）经常返回带有"Context size has been exceeded"等消息的 HTTP 400，
                    # 这必须触发上下文压缩，而不是立即中止。
                    is_context_length_error = any(phrase in error_msg for phrase in [
                        'context length', 'context size', 'maximum context',
                        'token limit', 'too many tokens', 'reduce the length',
                        'exceeds the limit', 'context window',
                        'request entity too large',  # OpenRouter/Nous 413 安全网
                        'prompt is too long',  # Anthropic: "prompt is too long: N tokens > M maximum"
                        'prompt exceeds max length',  # Z.AI / GLM: 通用 400 溢出用词
                    ])

                    # 回退启发式：当上下文太大时，Anthropic 有时会返回带有"Error"消息的通用 400 invalid_request_error。
                    # 如果错误消息非常短且会话很大，将其视为可能的上下文长度错误并尝试压缩。
                    if not is_context_length_error and status_code == 400:
                        ctx_len = getattr(getattr(self, 'context_compressor', None), 'context_length', 200000)
                        is_large_session = approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
                        is_generic_error = len(error_msg.strip()) < 30  # 例如仅 "error"
                        if is_large_session and is_generic_error:
                            is_context_length_error = True
                            self._vprint(
                                f"{self.log_prefix}⚠️  Generic 400 with large session "
                                f"(~{approx_tokens:,} tokens, {len(api_messages)} msgs) — "
                                f"treating as probable context overflow.",
                                force=True,
                            )

                    # 大会话上的服务器断开连接通常是由请求超过上下文/payload 限制而没有正确响应引起的。
                    # 将这些视为上下文长度错误以触发压缩，而不是消耗注定失败的重试。
                    if not is_context_length_error and not status_code:
                        _is_server_disconnect = (
                            'server disconnected' in error_msg
                            or 'peer closed connection' in error_msg
                            or error_type in ('ReadError', 'RemoteProtocolError', 'ServerDisconnectedError')
                        )
                        if _is_server_disconnect:
                            ctx_len = getattr(getattr(self, 'context_compressor', None), 'context_length', 200000)
                            _is_large = approx_tokens > ctx_len * 0.6 or len(api_messages) > 200
                            if _is_large:
                                is_context_length_error = True
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Server disconnected with large session "
                                    f"(~{approx_tokens:,} tokens, {len(api_messages)} msgs) — "
                                    f"treating as context-length error, attempting compression.",
                                    force=True,
                                )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # 尝试从错误消息中解析实际限制
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                        else:
                            # 下一层探测层级
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.context_length = new_ctx
                            compressor.threshold_tokens = int(new_ctx * compressor.threshold_percent)
                            compressor._context_probed = True
                            # 只持久化从提供商错误消息中解析的限制（真实数字）。
                            compressor._context_probe_persistable = bool(
                                parsed_limit and parsed_limit == new_ctx
                            )
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True
                            }
                        self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # 压缩重试之间的短暂暂停
                            restart_with_compressed_messages = True
                            break
                        else:
                            # 无法进一步压缩且已处于最低层级
                            self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True
                            }

                    # 检查不可重试的客户端错误（4xx HTTP 状态码）。
                    _RETRYABLE_STATUS_CODES = {413, 429, 529}
                    is_local_validation_error = (
                        isinstance(api_error, (ValueError, TypeError))
                        and not isinstance(api_error, UnicodeEncodeError)
                    )
                    # 检测来自 Anthropic OAuth 的通用 400（瞬态服务器端失败）。
                    # Real invalid_request_error responses include a descriptive message;
                    # transient ones contain only "Error" or are empty. (ref: issue #1608)
                    _err_body = getattr(api_error, "body", None) or {}
                    _err_message = (_err_body.get("error", {}).get("message", "") if isinstance(_err_body, dict) else "")
                    _is_generic_400 = (status_code == 400 and _err_message.strip().lower() in ("error", ""))
                    is_client_status_error = isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in _RETRYABLE_STATUS_CODES and not _is_generic_400
                    is_client_error = (is_local_validation_error or is_client_status_error or any(phrase in error_msg for phrase in [
                        'error code: 401', 'error code: 403',
                        'error code: 404', 'error code: 422',
                        'is not a valid model', 'invalid model', 'model not found',
                        'invalid api key', 'invalid_api_key', 'authentication',
                        'unauthorized', 'forbidden', 'not found',
                    ])) and not is_context_length_error

                    if is_client_error:
                        # Try fallback before aborting — a different provider
                        # may not have the same issue (rate limit, auth, etc.)
                        self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        self._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                        self._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{self._summarize_api_error(api_error)}"
                        )
                        self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                        self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                        self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                        if status_code in (401, 403) or "unauthorized" in error_msg or "forbidden" in error_msg or "permission" in error_msg:
                            if _provider == "openai-codex" and status_code == 401:
                                self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            else:
                                self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                                self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                                self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                                if "openrouter" in str(_base).lower():
                                    self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")

                        # 当错误疑似上下文溢出（400 状态码且会话过长）时，跳过会话落盘。
                        # 避免固化失败的消息导致下次请求陷入同样的循环。 (#1630)
                        if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Skipping session persistence "
                                f"for large failed session to prevent growth loop.",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # 在回退之前，尝试针对瞬时传输错误（陈旧连接池、TCP 重置）重新构建客户端。
                        if not primary_recovery_attempted and self._try_recover_primary_transport(
                            api_error, retry_count=retry_count, max_retries=max_retries,
                        ):
                            primary_recovery_attempted = True
                            retry_count = 0
                            continue
                        
                        self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        _final_summary = self._summarize_api_error(api_error)
                        if is_rate_limited:
                            self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                        else:
                            self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                        self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

                        # 检测 SSE 流中断模式（例如“网络连接丢失”）并提供引导。
                        # 这通常发生在模型生成超大 Tool Call 时，代理/CDN 中途断开流。
                        _is_stream_drop = (
                            not getattr(api_error, "status_code", None)
                            and any(p in error_msg for p in (
                                "connection lost", "connection reset",
                                "connection closed", "network connection",
                                "network error", "terminated",
                            ))
                        )
                        if _is_stream_drop:
                            self._vprint(
                                f"{self.log_prefix}   💡 The provider's stream "
                                f"connection keeps dropping. This often happens "
                                f"when the model tries to write a very large "
                                f"file in a single tool call.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try asking the model "
                                f"to use execute_code with Python's open() for "
                                f"large files, or to write the file in smaller "
                                f"sections.",
                                force=True,
                            )

                        logging.error(
                            "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                            self.log_prefix, max_retries, _final_summary,
                            _provider, _model, len(api_messages), f"{approx_tokens:,}",
                        )
                        self._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                        self._persist_session(messages, conversation_history)
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                        if _is_stream_drop:
                            _final_response += (
                                "\n\nThe provider's stream connection keeps "
                                "dropping — this often happens when generating "
                                "very large tool call responses (e.g. write_file "
                                "with long content). Try asking me to use "
                                "execute_code with Python's open() for large "
                                "files, or to write in smaller sections."
                            )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _final_summary,
                        }

                    _retry_after = None
                    if is_rate_limited:
                        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                        if _resp_headers and hasattr(_resp_headers, "get"):
                            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                            if _ra_raw:
                                try:
                                    _retry_after = min(int(_ra_raw), 120)  # Cap at 2 minutes
                                except (TypeError, ValueError):
                                    pass
                    wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                    if is_rate_limited:
                        self._emit_status(f"⏱️ Rate limit reached. Waiting {wait_time}s before retry (attempt {retry_count + 1}/{max_retries})...")
                    else:
                        self._emit_status(f"⏳ Retrying in {wait_time}s (attempt {retry_count}/{max_retries})...")
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    
                    sleep_end = time.time() + wait_time
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)
            
            if interrupted:
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # 将上下文压缩导致的重启计入重试次数，防止在压缩效果不足时陷入死循环。
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                continue

            # 如果所有重试均告失败（例如重复的上下文长度错误），退出循环。
            if response is None:
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                if self.api_mode == "codex_responses":
                    assistant_message, finish_reason = self._normalize_codex_response(response)
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=self._is_anthropic_oauth
                    )
                else:
                    assistant_message = response.choices[0].message
                
                # 标准化内容为字符串，防止部分兼容服务器返回 dict 或 list 导致 downstream 崩溃。
                if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                    elif isinstance(raw, list):
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    _assistant_text = assistant_message.content or ""
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        usage=self._usage_summary_for_api_request_hook(response),
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
                except Exception:
                    pass

                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # 将模型思考过程通知进度回调（用于子代理向父代理透传推理过程）。
                if (assistant_message.content and self.tool_progress_callback):
                    _think_text = assistant_message.content.strip()
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line and getattr(self, '_delegate_depth', 0) > 0:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                    elif _think_text:
                        try:
                            self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                        except Exception:
                            pass
                
                # 检测不完整的 <REASONING_SCRATCHPAD>（有始无终），通常意味着输出被截断，重试最多 2 次。
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    if not hasattr(self, '_incomplete_scratchpad_retries'):
                        self._incomplete_scratchpad_retries = 0
                    self._incomplete_scratchpad_retries += 1
                    
                    self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        continue
                    else:
                        self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        
                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                        }
                
                if hasattr(self, '_incomplete_scratchpad_retries'):
                    self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    if not hasattr(self, "_codex_incomplete_retries"):
                        self._codex_incomplete_retries = 0
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

                    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                        last_msg = messages[-1] if messages else None
                        # 重复检测：合并内容和推理均完全相同的连续不完整回复。
                        last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0
                
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # 验证 Tool Call 名称，修复模型幻觉。
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        if not hasattr(self, '_invalid_tool_retries'):
                            self._invalid_tool_retries = 0
                        self._invalid_tool_retries += 1

                        # 向模型返回有用的错误信息，以便其在下一轮自行纠正。
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                        self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                        if self._invalid_tool_retries >= 3:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }

                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    
                    if hasattr(self, '_invalid_tool_retries'):
                        self._invalid_tool_retries = 0
                    
                    # 验证 Tool Call 参数是否为有效的 JSON，并处理模型常见的空字符串等异常。
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        self._invalid_json_retries += 1
                        
                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")
                        
                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            continue
                        else:
                            # 超过重试次数后，注入 Tool Error 结果以便模型尝试恢复。
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                            self._invalid_json_retries = 0
                            
                            recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(recovery_assistant)
                            
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            continue
                    
                    self._invalid_json_retries = 0

                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # 如果当前轮次同时包含 Content 和 Tool Calls，则捕获 Content 作为降级回复。
                    # 场景：模型在回答的同时触发了 Memory/Skill 等副作用工具。
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # 如果当前轮次的所有 Tool Call 都是后台维护类（Memory, Todo 等），
                        # 且存在流式消费者，则静默后续输出。
                        _HOUSEKEEPING_TOOLS = frozenset({
                            "memory", "todo", "skill_manage", "session_search",
                        })
                        _all_housekeeping = all(
                            tc.function.name in _HOUSEKEEPING_TOOLS
                            for tc in assistant_message.tool_calls
                        )
                        if _all_housekeeping and self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self.quiet_mode:
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    # 在追加 Tool Call 路径的消息前，弹出 thinking-only prefill 消息。
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()

                    messages.append(assistant_msg)

                    # 在执行工具前，关闭任何打开的流式显示框。
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    # 标记下次流式文本到达时需要插入分段符。
                    self._stream_needs_break = True

                    # 如果仅执行了 execute_code（编程式调用），则返还迭代配额。
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # 使用 API 返回的真实 Token 计数来决定是否触发上下文压缩。
                    _compressor = self.context_compressor
                    if _compressor.last_prompt_tokens > 0:
                        _real_tokens = (
                            _compressor.last_prompt_tokens
                            + _compressor.last_completion_tokens
                        )
                    else:
                        _real_tokens = estimate_messages_tokens_rough(messages)

                    # 上下文压力预警（仅面向用户）。
                    if _compressor.threshold_tokens > 0:
                        _compaction_progress = _real_tokens / _compressor.threshold_tokens
                        if _compaction_progress >= 0.85 and not self._context_pressure_warned:
                            self._context_pressure_warned = True
                            self._emit_context_pressure(_compaction_progress, _compressor)

                    if self.compression_enabled and _compressor.should_compress(_real_tokens):
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                        conversation_history = None
                    
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    continue
                
                else:
                    final_response = assistant_message.content or ""
                    
                    if not self._has_content_after_think_block(final_response):
                        # 如果前一轮已经在 Tool Call 的同时交付了真实内容，直接使用该内容。
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback:
                            logger.debug("Empty follow-up after tool calls — using prior turn content as final response")
                            self._last_content_with_tools = None
                            self._empty_content_retries = 0
                            for i in range(len(messages) - 1, -1, -1):
                                msg = messages[i]
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    tool_names = []
                                    for tc in msg["tool_calls"]:
                                        if not tc or not isinstance(tc, dict): continue
                                        fn = tc.get("function", {})
                                        tool_names.append(fn.get("name", "unknown"))
                                    msg["content"] = f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                    break
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # Thinking-only prefill 延续逻辑：
                        # 模型生成了结构化推理但没有正文内容时，追加该消息并继续请求，
                        # 模型将在下一轮看到自己的推理并补全正文。
                        _has_structured = bool(
                            getattr(assistant_message, "reasoning", None)
                            or getattr(assistant_message, "reasoning_content", None)
                            or getattr(assistant_message, "reasoning_details", None)
                        )
                        if _has_structured and self._thinking_prefill_retries < 2:
                            self._thinking_prefill_retries += 1
                            self._vprint(
                                f"{self.log_prefix}↻ Thinking-only response — "
                                f"prefilling to continue "
                                f"({self._thinking_prefill_retries}/2)"
                            )
                            interim_msg = self._build_assistant_message(
                                assistant_message, "incomplete"
                            )
                            interim_msg["_thinking_prefill"] = True
                            messages.append(interim_msg)
                            self._session_messages = messages
                            self._save_session_log(messages)
                            continue

                        # Exhausted prefill attempts or no structured
                        # reasoning — fall through to "(empty)" terminal.
                        reasoning_text = self._extract_reasoning(assistant_message)
                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        assistant_msg["content"] = "(empty)"
                        messages.append(assistant_msg)

                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            self._vprint(f"{self.log_prefix}ℹ️  Reasoning-only response (no visible content). Reasoning: {reasoning_preview}")
                        else:
                            self._vprint(f"{self.log_prefix}ℹ️  Empty response (no content or reasoning).")

                        final_response = "(empty)"
                        break
                    
                    # Reset retry counter/signature on successful content
                    if hasattr(self, '_empty_content_retries'):
                        self._empty_content_retries = 0
                    self._last_empty_content_signature = None
                    self._thinking_prefill_retries = 0

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                        messages.append(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    if truncated_response_prefix:
                        final_response = truncated_response_prefix + final_response
                        truncated_response_prefix = ""
                        length_continue_retries = 0
                    
                    final_response = self._strip_think_blocks(final_response).strip()
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)

                    # 在追加最终回复前，弹出 thinking-only prefill 消息，保持历史整洁。
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()

                    messages.append(final_msg)
                    
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                if self.verbose_logging:
                    logging.exception("Detailed error information:")
                
                # 如果 Assistant 消息已包含 Tool Calls 但发生错误，为所有未处理的 ID 填充错误结果。
                pending_handled = False
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                    break
                
                # 非工具错误无需注入合成消息，直接报错并继续重试，避免污染历史记录。

                if api_call_count >= self.max_iterations - 1:
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            if self.iteration_budget.remaining <= 0 and not self.quiet_mode:
                print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        self._save_trajectory(messages, user_message, final_response is not None and api_call_count < self.max_iterations)

        self._cleanup_task_resources(effective_task_id)

        # 会话落盘：保存到 JSON 日志和 SQLite 数据库。
        self._persist_session(messages, conversation_history)

        # 插件钩子：LLM 调用结束后触发。
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": final_response is not None and api_call_count < self.max_iterations,
            "partial": False,
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        self._response_was_previewed = False
        
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        self.clear_interrupt()
        self._stream_callback = None

        _should_review_skills = False
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            self._iters_since_skill = 0

        # 外部 Memory Provider：同步当前轮次并为下一轮预取。
        if self._memory_manager and final_response and original_user_message:
            try:
                self._memory_manager.sync_all(original_user_message, final_response)
                self._memory_manager.queue_prefetch_all(original_user_message)  # 【文档锚点 5A】主响应结束后为下一轮预热 memory recall
            except Exception:
                pass

        # 后台复盘：异步触发 Memory/Skill Review。
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass

        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=result["completed"],
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """
        Simple chat interface that returns just the final response.

        Args:
            message (str): User message
            stream_callback: Optional callback invoked with each text delta during streaming.

        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-sonnet-4.6.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection. Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses. Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    if list_tools:
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 50)
        
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")
        
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")
        
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")
        
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")
    
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"
        
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")



if __name__ == "__main__":
    fire.Fire(main)
