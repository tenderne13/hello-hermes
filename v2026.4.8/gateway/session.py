"""
Gateway 会话管理模块。

职责：
- 会话上下文追踪（消息来源）
- 会话存储（对话持久化到磁盘）
- 重置策略评估（何时开始新会话）
- 动态系统提示词注入（agent 感知自身上下文）
"""

import hashlib
import logging
import os
import json
import re
import threading
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _now() -> datetime:
    """返回当前本地时间。"""
    return datetime.now()


# ---------------------------------------------------------------------------
# PII 数据脱敏辅助函数
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"^\+?\d[\d\-\s]{6,}$")


def _hash_id(value: str) -> str:
    """返回标识符的确定性 12 字符十六进制哈希。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _hash_sender_id(value: str) -> str:
    """将发送者 ID 哈希为 ``user_<12位十六进制>`` 格式。"""
    return f"user_{_hash_id(value)}"


def _hash_chat_id(value: str) -> str:
    """哈希聊天 ID 的数字部分，保留平台前缀。

    ``telegram:12345`` → ``telegram:<hash>``
    ``12345``          → ``<hash>``
    """
    colon = value.find(":")
    if colon > 0:
        prefix = value[:colon]
        return f"{prefix}:{_hash_id(value[colon + 1:])}"
    return _hash_id(value)


def _looks_like_phone(value: str) -> bool:
    """如果 *value* 看起来像电话号码（E.164 或类似格式），返回 True。"""
    return bool(_PHONE_RE.match(value.strip()))

from .config import (
    Platform,
    GatewayConfig,
    SessionResetPolicy,  # noqa: F401 — re-exported via gateway/__init__.py
    HomeChannel,
)


@dataclass
class SessionSource:
    """
    描述消息的来源。

    此信息用于：
    1. 将响应路由到正确的位置
    2. 注入系统提示词的上下文
    3. 追踪定时任务的发送来源
    """
    platform: Platform
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"  # "dm", "group", "channel", "thread"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_id: Optional[str] = None  # For forum topics, Discord threads, etc.
    chat_topic: Optional[str] = None  # Channel topic/description (Discord, Slack)
    user_id_alt: Optional[str] = None  # Signal UUID (alternative to phone number)
    chat_id_alt: Optional[str] = None  # Signal group internal ID

    @property
    def description(self) -> str:
        """返回人类可读的来源描述。"""
        if self.platform == Platform.LOCAL:
            return "CLI terminal"

        parts = []
        if self.chat_type == "dm":
            parts.append(f"DM with {self.user_name or self.user_id or 'user'}")
        elif self.chat_type == "group":
            parts.append(f"group: {self.chat_name or self.chat_id}")
        elif self.chat_type == "channel":
            parts.append(f"channel: {self.chat_name or self.chat_id}")
        else:
            parts.append(self.chat_name or self.chat_id)

        if self.thread_id:
            parts.append(f"thread: {self.thread_id}")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "thread_id": self.thread_id,
            "chat_topic": self.chat_topic,
        }
        if self.user_id_alt:
            d["user_id_alt"] = self.user_id_alt
        if self.chat_id_alt:
            d["chat_id_alt"] = self.chat_id_alt
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSource":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            chat_name=data.get("chat_name"),
            chat_type=data.get("chat_type", "dm"),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            thread_id=data.get("thread_id"),
            chat_topic=data.get("chat_topic"),
            user_id_alt=data.get("user_id_alt"),
            chat_id_alt=data.get("chat_id_alt"),
        )

    @classmethod
    def local_cli(cls) -> "SessionSource":
        """创建代表本地 CLI 的来源。"""
        return cls(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI terminal",
            chat_type="dm",
        )


@dataclass
class SessionContext:
    """
    完整会话上下文，用于动态系统提示词注入。

    Agent 接收此信息以了解：
    - 消息来自何处
    - 有哪些平台可用
    - 可以向何处发送定时任务输出
    """
    source: SessionSource
    connected_platforms: List[Platform]
    home_channels: Dict[Platform, HomeChannel]

    # Session metadata
    session_key: str = ""
    session_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "connected_platforms": [p.value for p in self.connected_platforms],
            "home_channels": {
                p.value: hc.to_dict() for p, hc in self.home_channels.items()
            },
            "session_key": self.session_key,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


_PII_SAFE_PLATFORMS = frozenset({
    Platform.WHATSAPP,
    Platform.SIGNAL,
    Platform.TELEGRAM,
})
"""用户 ID 可安全脱敏的平台（没有需要原始 ID 的内置提及系统）。
排除 Discord，因为提及使用 ``<@user_id>`` 格式，LLM 需要真实 ID 才能标记用户。"""


def build_session_context_prompt(
    context: SessionContext,
    *,
    redact_pii: bool = False,
) -> str:
    """
    构建告诉 agent 其上下文的动态系统提示词部分。

    此信息注入到系统提示词中，使 agent 知道：
    - 消息来自何处
    - 已连接哪些平台
    - 可以向何处发送定时任务输出

    当 *redact_pii* 为 True **且** 来源平台在
    ``_PII_SAFE_PLATFORMS`` 中时，电话号码被剥离，用户/聊天 ID
    在发送给 LLM 前被替换为确定性哈希。
    Discord 等平台被排除，因为提及需要真实 ID。
    路由仍使用原始值（它们保留在 SessionSource 中）。
    """
    # 仅在需要 ID 进行提及的平台应用脱敏
    redact_pii = redact_pii and context.source.platform in _PII_SAFE_PLATFORMS
    lines = [
        "## Current Session Context",
        "",
    ]

    # 平台信息
    platform_name = context.source.platform.value.title()
    if context.source.platform == Platform.LOCAL:
        lines.append(f"**Source:** {platform_name} (the machine running this agent)")
    else:
        # 构建尊重 PII 脱敏的描述
        src = context.source
        if redact_pii:
            # 构建不含原始 ID 的安全描述
            _uname = src.user_name or (
                _hash_sender_id(src.user_id) if src.user_id else "user"
            )
            _cname = src.chat_name or _hash_chat_id(src.chat_id)
            if src.chat_type == "dm":
                desc = f"DM with {_uname}"
            elif src.chat_type == "group":
                desc = f"group: {_cname}"
            elif src.chat_type == "channel":
                desc = f"channel: {_cname}"
            else:
                desc = _cname
        else:
            desc = src.description
        lines.append(f"**Source:** {platform_name} ({desc})")

    # 频道主题（如果可用 - 提供关于频道用途的上下文）
    if context.source.chat_topic:
        lines.append(f"**Channel Topic:** {context.source.chat_topic}")

    # 用户身份
    # 在共享线程会话中（非 DM 且有 thread_id），多个用户
    # 参与同一对话。不要在系统提示词中固定单个用户名，
    # 因为每轮都会变化，会破坏提示词缓存。
    # 应注意这是多用户线程；各个发送者名称由 gateway
    # 添加在每条用户消息的前缀中。
    _is_shared_thread = (
        context.source.chat_type != "dm"
        and context.source.thread_id
    )
    if _is_shared_thread:
        lines.append(
            "**Session type:** Multi-user thread — messages are prefixed "
            "with [sender name]. Multiple users may participate."
        )
    elif context.source.user_name:
        lines.append(f"**User:** {context.source.user_name}")
    elif context.source.user_id:
        uid = context.source.user_id
        if redact_pii:
            uid = _hash_sender_id(uid)
        lines.append(f"**User ID:** {uid}")

    # 平台特定行为说明
    if context.source.platform == Platform.SLACK:
        lines.append("")
        lines.append(
            "**Platform notes:** You are running inside Slack. "
            "You do NOT have access to Slack-specific APIs — you cannot search "
            "channel history, pin/unpin messages, manage channels, or list users. "
            "Do not promise to perform these actions. If the user asks, explain "
            "that you can only read messages sent directly to you and respond."
        )
    elif context.source.platform == Platform.DISCORD:
        lines.append("")
        lines.append(
            "**Platform notes:** You are running inside Discord. "
            "You do NOT have access to Discord-specific APIs — you cannot search "
            "channel history, pin messages, manage roles, or list server members. "
            "Do not promise to perform these actions. If the user asks, explain "
            "that you can only read messages sent directly to you and respond."
        )

    # 已连接的平台列表
    platforms_list = ["local (files on this machine)"]
    for p in context.connected_platforms:
        if p != Platform.LOCAL:
            platforms_list.append(f"{p.value}: Connected ✓")

    lines.append(f"**Connected Platforms:** {', '.join(platforms_list)}")

    # 主频道
    if context.home_channels:
        lines.append("")
        lines.append("**Home Channels (default destinations):**")
        for platform, home in context.home_channels.items():
            hc_id = _hash_chat_id(home.chat_id) if redact_pii else home.chat_id
            lines.append(f"  - {platform.value}: {home.name} (ID: {hc_id})")

    # 定时任务投递选项
    lines.append("")
    lines.append("**Delivery options for scheduled tasks:**")

    # 来源投递
    if context.source.platform == Platform.LOCAL:
        lines.append("- `\"origin\"` → Local output (saved to files)")
    else:
        _origin_label = context.source.chat_name or (
            _hash_chat_id(context.source.chat_id) if redact_pii else context.source.chat_id
        )
        lines.append(f"- `\"origin\"` → Back to this chat ({_origin_label})")

    # 本地始终可用
    lines.append("- `\"local\"` → Save to local files only (~/.hermes/cron/output/)")

    # 平台主频道
    for platform, home in context.home_channels.items():
        lines.append(f"- `\"{platform.value}\"` → Home channel ({home.name})")

    # 显式定位说明
    lines.append("")
    lines.append("*For explicit targeting, use `\"platform:chat_id\"` format if the user provides a specific chat ID.*")

    return "\n".join(lines)


@dataclass
class SessionEntry:
    """
    会话存储中的条目。

    将会话键映射到其当前会话 ID 和元数据。
    """
    session_key: str
    session_id: str
    created_at: datetime
    updated_at: datetime

    # 用于投递路由的来源元数据
    origin: Optional[SessionSource] = None

    # 显示元数据
    display_name: Optional[str] = None
    platform: Optional[Platform] = None
    chat_type: str = "dm"

    # Token 追踪
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_status: str = "unknown"

    # 上次 API 报告的提示词 token 数（用于准确的压缩预检查）
    last_prompt_tokens: int = 0

    # 当会话因前一个过期而创建时设置；
    # 被消息处理器消费一次以注入上下文通知
    was_auto_reset: bool = False
    auto_reset_reason: Optional[str] = None  # "idle" or "daily"
    reset_had_activity: bool = False  # whether the expired session had any messages

    # 由后台过期监视器在成功刷新
    # 此会话的记忆后设置。持久化到 sessions.json 以使标志
    # 在 gateway 重启后保留（旧的内存中的 _pre_flushed_sessions
    # 集在重启时丢失，导致冗余的重新刷新）。
    memory_flushed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_key": self.session_key,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "display_name": self.display_name,
            "platform": self.platform.value if self.platform else None,
            "chat_type": self.chat_type,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_status": self.cost_status,
            "memory_flushed": self.memory_flushed,
        }
        if self.origin:
            result["origin"] = self.origin.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionEntry":
        origin = None
        if "origin" in data and data["origin"]:
            origin = SessionSource.from_dict(data["origin"])

        platform = None
        if data.get("platform"):
            try:
                platform = Platform(data["platform"])
            except ValueError as e:
                logger.debug("Unknown platform value %r: %s", data["platform"], e)

        return cls(
            session_key=data["session_key"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            origin=origin,
            display_name=data.get("display_name"),
            platform=platform,
            chat_type=data.get("chat_type", "dm"),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            last_prompt_tokens=data.get("last_prompt_tokens", 0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
            cost_status=data.get("cost_status", "unknown"),
            memory_flushed=data.get("memory_flushed", False),
        )


def build_session_key(
    source: SessionSource,
    group_sessions_per_user: bool = True,
    thread_sessions_per_user: bool = False,
) -> str:
    """从消息来源构建确定性会话键。

    这是会话键构造的唯一权威来源。

    DM 规则：
      - DMs 在有 chat_id 时包含它，因此每个私人对话是隔离的。
      - thread_id 进一步区分同一 DM 聊天中的线程 DM。
      - 没有 chat_id 时，thread_id 作为尽力而为的备选。
      - 没有 thread_id 或 chat_id 时，DMs 共享单个会话。

    群组/频道规则：
      - chat_id 标识父群组/频道。
      - 当 ``group_sessions_per_user`` 启用时，user_id/user_id_alt 在有可用时隔离该父聊天中的参与者。
      - thread_id 区分该父聊天中的线程。当
        ``thread_sessions_per_user`` 为 False（默认）时，线程在所有
        参与者间*共享* — user_id 不附加，因此线程中的每个用户
        共享一个会话。这是线程化对话的预期 UX
        （Telegram 论坛话题、Discord 线程、Slack 线程）。
      - 没有参与者标识符，或隔离被禁用时，消息退回到每个聊天一个
        共享会话。
      - 没有标识符时，消息退回到每个平台/聊天类型一个会话。
    """
    platform = source.platform.value
    if source.chat_type == "dm":
        if source.chat_id:
            if source.thread_id:
                return f"agent:main:{platform}:dm:{source.chat_id}:{source.thread_id}"
            return f"agent:main:{platform}:dm:{source.chat_id}"
        if source.thread_id:
            return f"agent:main:{platform}:dm:{source.thread_id}"
        return f"agent:main:{platform}:dm"

    participant_id = source.user_id_alt or source.user_id
    key_parts = ["agent:main", platform, source.chat_type]

    if source.chat_id:
        key_parts.append(source.chat_id)
    if source.thread_id:
        key_parts.append(source.thread_id)

    # 在线程中，默认使用共享会话（所有参与者看到相同的
    # 对话）。仅在通过 thread_sessions_per_user 明确启用时
    # 或在没有线程（普通群组）时应用每用户隔离。
    isolate_user = group_sessions_per_user
    if source.thread_id and not thread_sessions_per_user:
        isolate_user = False

    if isolate_user and participant_id:
        key_parts.append(str(participant_id))

    return ":".join(key_parts)


# ============================================================
# SessionStore - 会话存储管理
# 职责：管理会话的存储和检索
# 使用 SQLite（通过 SessionDB）存储会话元数据和消息记录
# 当 SQLite 不可用时回退到遗留的 JSONL 文件
# 相关方法：append_to_transcript()、rewrite_transcript()、load_transcript()
# ============================================================
class SessionStore:

    def __init__(self, sessions_dir: Path, config: GatewayConfig,
                 has_active_processes_fn=None,
                 on_auto_reset=None):
        self.sessions_dir = sessions_dir
        self.config = config
        self._entries: Dict[str, SessionEntry] = {}
        self._loaded = False
        self._lock = threading.Lock()
        self._has_active_processes_fn = has_active_processes_fn

        # 初始化 SQLite 会话数据库
        self._db = None
        try:
            from hermes_state import SessionDB
            self._db = SessionDB()
        except Exception as e:
            print(f"[gateway] Warning: SQLite session store unavailable, falling back to JSONL: {e}")

    def _ensure_loaded(self) -> None:
        """如果会话索引尚未加载，则从磁盘加载。"""
        with self._lock:
            self._ensure_loaded_locked()

    def _ensure_loaded_locked(self) -> None:
        """从磁盘加载会话索引。必须在持有 self._lock 时调用。"""
        if self._loaded:
            return

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_file = self.sessions_dir / "sessions.json"

        if sessions_file.exists():
            try:
                with open(sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        try:
                            self._entries[key] = SessionEntry.from_dict(entry_data)
                        except (ValueError, KeyError):
                            # 跳过包含未知/已移除平台值的条目
                            continue
            except Exception as e:
                print(f"[gateway] Warning: Failed to load sessions: {e}")

        self._loaded = True

    def _save(self) -> None:
        """将会话索引保存到磁盘（保留用于会话键 -> ID 映射）。"""
        import tempfile
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_file = self.sessions_dir / "sessions.json"

        data = {key: entry.to_dict() for key, entry in self._entries.items()}
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.sessions_dir), suffix=".tmp", prefix=".sessions_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, sessions_file)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.debug("Could not remove temp file %s: %s", tmp_path, e)
            raise

    def _generate_session_key(self, source: SessionSource) -> str:
        """从来源生成会话键。"""
        return build_session_key(
            source,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )

    def _is_session_expired(self, entry: SessionEntry) -> bool:
        """根据重置策略检查会话是否已过期。

        仅从条目工作 — 无需 SessionSource。
        由后台过期监视器用于主动刷新记忆。
        有活跃后台进程的会话永远不会被视为过期。
        """
        if self._has_active_processes_fn:
            if self._has_active_processes_fn(entry.session_key):
                return False

        policy = self.config.get_reset_policy(
            platform=entry.platform,
            session_type=entry.chat_type,
        )

        if policy.mode == "none":
            return False

        now = _now()

        if policy.mode in ("idle", "both"):
            idle_deadline = entry.updated_at + timedelta(minutes=policy.idle_minutes)
            if now > idle_deadline:
                return True

        if policy.mode in ("daily", "both"):
            today_reset = now.replace(
                hour=policy.at_hour,
                minute=0, second=0, microsecond=0,
            )
            if now.hour < policy.at_hour:
                today_reset -= timedelta(days=1)
            if entry.updated_at < today_reset:
                return True

        return False

    def _should_reset(self, entry: SessionEntry, source: SessionSource) -> Optional[str]:
        """
        根据策略检查会话是否应重置。

        如果需要重置，返回重置原因（"idle" 或 "daily"），
        如果会话仍有效则返回 None。

        有活跃后台进程的会话永远不会被重置。
        """
        if self._has_active_processes_fn:
            session_key = self._generate_session_key(source)
            if self._has_active_processes_fn(session_key):
                return None

        policy = self.config.get_reset_policy(
            platform=source.platform,
            session_type=source.chat_type
        )

        if policy.mode == "none":
            return None

        now = _now()

        if policy.mode in ("idle", "both"):
            idle_deadline = entry.updated_at + timedelta(minutes=policy.idle_minutes)
            if now > idle_deadline:
                return "idle"

        if policy.mode in ("daily", "both"):
            today_reset = now.replace(
                hour=policy.at_hour,
                minute=0,
                second=0,
                microsecond=0
            )
            if now.hour < policy.at_hour:
                today_reset -= timedelta(days=1)

            if entry.updated_at < today_reset:
                return "daily"

        return None

    def has_any_sessions(self) -> bool:
        """检查是否创建过任何会话（跨所有平台）。

        使用 SQLite 数据库作为真值来源，因为它保留
        历史会话记录（已结束的会话仍被计算）。内存中的
        ``_entries`` 字典在重置时替换条目，所以 ``len(_entries)`` 对于单平台用户保持为 1 — 这是此方法修复的 bug。

        当前会话在调用此方法时已在 DB 中
        （get_or_create_session 首先运行），所以我们检查 ``> 1``。
        """
        if self._db:
            try:
                return self._db.session_count() > 1
            except Exception:
                pass  # 降级到启发式判断
        # 备选：检查 sessions.json 是否加载了现有数据。
        # 这覆盖了 DB 不可用的罕见情况。
        with self._lock:
            self._ensure_loaded_locked()
            return len(self._entries) > 1

    def get_or_create_session(
        self,
        source: SessionSource,
        force_new: bool = False
    ) -> SessionEntry:
        """
        获取现有会话或创建新会话。

        评估重置策略以确定现有会话是否过期。
        在新会话开始时在 SQLite 中创建会话记录。
        """
        session_key = self._generate_session_key(source)
        now = _now()

        # SQLite 调用在锁外进行以避免在 I/O 期间持有锁。
        # 所有 _entries / _loaded 变更受 self._lock 保护。
        db_end_session_id = None
        db_create_kwargs = None

        with self._lock:
            self._ensure_loaded_locked()

            if session_key in self._entries and not force_new:
                entry = self._entries[session_key]

                reset_reason = self._should_reset(entry, source)
                if not reset_reason:
                    entry.updated_at = now
                    self._save()
                    return entry
                else:
                    # 会话正在被自动重置。
                    was_auto_reset = True
                    auto_reset_reason = reset_reason
                    # 追踪过期的会话是否有真正的对话
                    reset_had_activity = entry.total_tokens > 0
                    db_end_session_id = entry.session_id
            else:
                was_auto_reset = False
                auto_reset_reason = None
                reset_had_activity = False

            # 创建新会话
            session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            entry = SessionEntry(
                session_key=session_key,
                session_id=session_id,
                created_at=now,
                updated_at=now,
                origin=source,
                display_name=source.chat_name,
                platform=source.platform,
                chat_type=source.chat_type,
                was_auto_reset=was_auto_reset,
                auto_reset_reason=auto_reset_reason,
                reset_had_activity=reset_had_activity,
            )

            self._entries[session_key] = entry
            self._save()
            db_create_kwargs = {
                "session_id": session_id,
                "source": source.platform.value,
                "user_id": source.user_id,
            }

        # 锁外的 SQLite 操作
        if self._db and db_end_session_id:
            try:
                self._db.end_session(db_end_session_id, "session_reset")
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)

        if self._db and db_create_kwargs:
            try:
                self._db.create_session(**db_create_kwargs)
            except Exception as e:
                print(f"[gateway] Warning: Failed to create SQLite session: {e}")

        # 用父 DM 会话历史种子化新的 DM 线程会话。
        # 当机器人回复创建 Slack 线程而用户在回复它时，
        # 线程获得一个新会话（由 thread_ts 键控）。如果没有种子化，
        # 线程会话以零上下文开始 — 用户的原始
        # 问题和对机器人的回答不可见。修复：将父
        # DM 会话的记录复制到新的线程会话中，使上下文延续，
        # 同时保持线程相互隔离。
        if (
            source.chat_type == "dm"
            and source.thread_id
            and entry.created_at == entry.updated_at  # brand-new session
            and not was_auto_reset
        ):
            parent_source = SessionSource(
                platform=source.platform,
                chat_id=source.chat_id,
                chat_type="dm",
                user_id=source.user_id,
                # no thread_id — this is the parent DM session
            )
            parent_key = self._generate_session_key(parent_source)
            with self._lock:
                parent_entry = self._entries.get(parent_key)
            if parent_entry and parent_entry.session_id != entry.session_id:
                try:
                    parent_history = self.load_transcript(parent_entry.session_id)
                    if parent_history:
                        self.rewrite_transcript(entry.session_id, parent_history)
                        logger.info(
                            "[Session] Seeded DM thread session %s with %d messages from parent %s",
                            entry.session_id, len(parent_history), parent_entry.session_id,
                        )
                except Exception as e:
                    logger.warning("[Session] Failed to seed thread session: %s", e)

        return entry

    def update_session(
        self,
        session_key: str,
        last_prompt_tokens: int = None,
    ) -> None:
        """在交互后更新轻量级会话元数据。"""
        with self._lock:
            self._ensure_loaded_locked()

            if session_key in self._entries:
                entry = self._entries[session_key]
                entry.updated_at = _now()
                if last_prompt_tokens is not None:
                    entry.last_prompt_tokens = last_prompt_tokens
                self._save()

    def reset_session(self, session_key: str) -> Optional[SessionEntry]:
        """强制重置会话，创建新会话 ID。"""
        db_end_session_id = None
        db_create_kwargs = None
        new_entry = None

        with self._lock:
            self._ensure_loaded_locked()

            if session_key not in self._entries:
                return None

            old_entry = self._entries[session_key]
            db_end_session_id = old_entry.session_id

            now = _now()
            session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            new_entry = SessionEntry(
                session_key=session_key,
                session_id=session_id,
                created_at=now,
                updated_at=now,
                origin=old_entry.origin,
                display_name=old_entry.display_name,
                platform=old_entry.platform,
                chat_type=old_entry.chat_type,
            )

            self._entries[session_key] = new_entry
            self._save()
            db_create_kwargs = {
                "session_id": session_id,
                "source": old_entry.platform.value if old_entry.platform else "unknown",
                "user_id": old_entry.origin.user_id if old_entry.origin else None,
            }

        if self._db and db_end_session_id:
            try:
                self._db.end_session(db_end_session_id, "session_reset")
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)

        if self._db and db_create_kwargs:
            try:
                self._db.create_session(**db_create_kwargs)
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)

        return new_entry

    def switch_session(self, session_key: str, target_session_id: str) -> Optional[SessionEntry]:
        """将会话键切换到指向现有会话 ID。

        由 ``/resume`` 用于恢复先前命名的会话。
        在 SQLite 中结束当前会话（如重置），但不是
        生成新的会话 ID，而是重用 ``target_session_id`` 以便
        在下一条消息时加载旧记录。
        """
        db_end_session_id = None
        new_entry = None

        with self._lock:
            self._ensure_loaded_locked()

            if session_key not in self._entries:
                return None

            old_entry = self._entries[session_key]

            # 如果已经在该会话上，则不切换
            if old_entry.session_id == target_session_id:
                return old_entry

            db_end_session_id = old_entry.session_id

            now = _now()
            new_entry = SessionEntry(
                session_key=session_key,
                session_id=target_session_id,
                created_at=now,
                updated_at=now,
                origin=old_entry.origin,
                display_name=old_entry.display_name,
                platform=old_entry.platform,
                chat_type=old_entry.chat_type,
            )

            self._entries[session_key] = new_entry
            self._save()

        if self._db and db_end_session_id:
            try:
                self._db.end_session(db_end_session_id, "session_switch")
            except Exception as e:
                logger.debug("Session DB end_session failed: %s", e)

        return new_entry

    def list_sessions(self, active_minutes: Optional[int] = None) -> List[SessionEntry]:
        """列出所有会话，可按活动筛选。"""
        with self._lock:
            self._ensure_loaded_locked()
            entries = list(self._entries.values())

        if active_minutes is not None:
            cutoff = _now() - timedelta(minutes=active_minutes)
            entries = [e for e in entries if e.updated_at >= cutoff]

        entries.sort(key=lambda e: e.updated_at, reverse=True)

        return entries

    def get_transcript_path(self, session_id: str) -> Path:
        """获取会话遗留记录文件的路径。"""
        # 【文档锚点 4C】Gateway transcript 边界：每个 session 仍保留 legacy JSONL 文件
        return self.sessions_dir / f"{session_id}.jsonl"

    def append_to_transcript(self, session_id: str, message: Dict[str, Any], skip_db: bool = False) -> None:
        """追加消息到会话记录（SQLite + 遗留 JSONL）。

        Args:
            skip_db: 为 True 时，仅写入 JSONL 并跳过 SQLite 写入。
                     用于当 agent 已通过自身的 _flush_messages_to_session_db()
                     持久化消息到 SQLite 时，避免重复写入 bug (#860)。
        """
        # 【文档锚点 4C】Gateway 写路径：同一条消息会写 SQLite，并兼容性追加 JSONL transcript
        # 写入 SQLite（除非 agent 已处理）
        if self._db and not skip_db:
            try:
                self._db.append_message(
                    session_id=session_id,
                    role=message.get("role", "unknown"),
                    content=message.get("content"),
                    tool_name=message.get("tool_name"),
                    tool_calls=message.get("tool_calls"),
                    tool_call_id=message.get("tool_call_id"),
                )
            except Exception as e:
                logger.debug("Session DB operation failed: %s", e)

        # 也写入遗留 JSONL（在过渡期间保持现有工具工作）
        transcript_path = self.get_transcript_path(session_id)
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    def rewrite_transcript(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """用新消息替换整个会话记录。

        用于 /retry、/undo 和 /compress 以持久化修改后的对话历史。
        重写 SQLite 和遗留 JSONL 存储。
        """
        # 【文档锚点 4C】Gateway 重写路径：/retry /undo /compress 会整体重写 transcript
        # SQLite：清除旧消息并重新插入
        if self._db:
            try:
                self._db.clear_messages(session_id)
                for msg in messages:
                    role = msg.get("role", "unknown")
                    self._db.append_message(
                        session_id=session_id,
                        role=role,
                        content=msg.get("content"),
                        tool_name=msg.get("tool_name"),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id"),
                        reasoning=msg.get("reasoning") if role == "assistant" else None,
                        reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                        codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                    )
            except Exception as e:
                logger.debug("Failed to rewrite transcript in DB: %s", e)

        # JSONL：覆盖文件
        transcript_path = self.get_transcript_path(session_id)
        with open(transcript_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def load_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        """从会话记录加载所有消息。"""
        # 【文档锚点 4C】Gateway 读取路径：优先返回 SQLite / JSONL 中历史更完整的一份
        db_messages = []
        # 首先尝试 SQLite
        if self._db:
            try:
                db_messages = self._db.get_messages_as_conversation(session_id)
            except Exception as e:
                logger.debug("Could not load messages from DB: %s", e)

        # 加载遗留 JSONL 记录（可能比 SQLite 包含更多历史，
        # 因为在 DB 层引入之前创建的会话）。
        transcript_path = self.get_transcript_path(session_id)
        jsonl_messages = []
        if transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            jsonl_messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping corrupt line in transcript %s: %s",
                                session_id, line[:120],
                            )

        # 优先使用消息更多的来源。
        #
        # 背景：当会话早于 SQLite 存储（或在长寿命会话已处于活动状态时
        # 添加了 DB 层），迁移后的第一轮只写入 *新* 消息到 SQLite
        # （因为 _flush_messages_to_session_db 跳过已存在
        # 于 conversation_history 中的消息，假设它们已被持久化）。
        # 在*下一*轮 load_transcript 返回那几条 SQLite 行而忽略
        # 完整的 JSONL 历史 — 模型看到 1-4 条消息的上下文而不是
        # 数百条。使用更长的来源可防止这种静默截断。
        if len(jsonl_messages) > len(db_messages):
            if db_messages:
                logger.debug(
                    "Session %s: JSONL has %d messages vs SQLite %d — "
                    "using JSONL (legacy session not yet fully migrated)",
                    session_id, len(jsonl_messages), len(db_messages),
                )
            return jsonl_messages

        return db_messages


def build_session_context(
    source: SessionSource,
    config: GatewayConfig,
    session_entry: Optional[SessionEntry] = None
) -> SessionContext:
    """
    从来源和配置构建完整会话上下文。

    用于将上下文注入 agent 的系统提示词。
    """
    connected = config.get_connected_platforms()

    home_channels = {}
    for platform in connected:
        home = config.get_home_channel(platform)
        if home:
            home_channels[platform] = home

    context = SessionContext(
        source=source,
        connected_platforms=connected,
        home_channels=home_channels,
    )

    if session_entry:
        context.session_key = session_entry.session_key
        context.session_id = session_entry.session_id
        context.created_at = session_entry.created_at
        context.updated_at = session_entry.updated_at

    return context
