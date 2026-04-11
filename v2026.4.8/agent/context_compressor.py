"""Automatic context window compression for long conversations.
自动上下文窗口压缩 — 用于长对话

Self-contained class with its own OpenAI client for summarization.
使用辅助模型 (廉价/快速) 总结中间轮次，同时保护头部和尾部上下文。

改进 v1:
  - 结构化摘要模板 (Goal, Progress, Decisions, Files, Next Steps)
  - 迭代摘要更新 (跨多次压缩保留信息)
  - Token 预算尾部保护而非固定消息数
  - LLM 摘要前的工具输出修剪 (廉价预传递)
  - 缩放摘要预算 (与压缩内容成比例)
  - 摘要器输入中更丰富的工具调用/结果详情
"""

import logging
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from agent.model_metadata import (
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted "
    "to save context space. The summary below describes work that was "
    "already completed, and the current session state may still reflect "
    "that work (for example, files may already be changed). Use the summary "
    "and the current state to continue from where things left off, and "
    "avoid repeating work:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# 摘要输出的最小 token 数
_MIN_SUMMARY_TOKENS = 2000
# 分配给摘要的压缩内容比例
_SUMMARY_RATIO = 0.20
# 摘要 token 的绝对上限
_SUMMARY_TOKENS_CEILING = 12_000

# 修剪旧工具结果时使用的占位符
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

_CHARS_PER_TOKEN = 4
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600


class ContextCompressor:
    """Compresses conversation context when approaching the model's context limit.
    当接近模型上下文限制时压缩对话上下文。

    算法:
      1. 修剪旧工具结果 (廉价，无需 LLM 调用)
      2. 保护头部消息 (系统提示 + 首次交换)
      3. 按 token 预算保护尾部消息 (最近约 20K tokens)
      4. 用结构化 LLM 提示总结中间轮次
      5. 后续压缩时，迭代更新前一个摘要

    压缩器会保留 _previous_summary，多次压缩时迭代更新摘要而不是每次重头总结。
    """

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        self.threshold_tokens = int(self.context_length * threshold_percent)
        self.compression_count = 0

        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        self._context_probed = False

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

        self.summary_model = summary_model_override or ""

        # 存储前一次压缩摘要，用于迭代更新
        self._previous_summary: Optional[str] = None  # 【文档锚点 4A】多次压缩不是重头总结，而是沿用上一次 handoff 摘要继续滚动
        self._summary_failure_cooldown_until: float = 0.0

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick pre-flight check using rough estimate (before API call)."""
        rough_estimate = estimate_messages_tokens_rough(messages)
        return rough_estimate >= self.threshold_tokens

    def get_status(self) -> Dict[str, Any]:
        """Get current compression status for display/logging."""
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": min(100, (self.last_prompt_tokens / self.context_length * 100)) if self.context_length else 0,
            "compression_count": self.compression_count,
        }

    # ------------------------------------------------------------------
    # 工具输出修剪 (廉价预传递，无需 LLM 调用)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Replace old tool result contents with a short placeholder.
        将旧的工具结果内容替换为短占位符。

        从末尾向后遍历，保护最近 ``protect_tail_count`` 条消息。
        较旧工具结果的内容被替换为占位符字符串。

        返回 (pruned_messages, pruned_count)。
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0
        prune_boundary = len(result) - protect_tail_count

        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            if len(content) > 200:
                result[i] = {**msg, "content": _PRUNED_TOOL_PLACEHOLDER}
                pruned += 1

        return result, pruned

    # ------------------------------------------------------------------
    # 摘要生成
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed.
        根据正在压缩的内容量缩放摘要 token 预算。

        最大值随模型上下文窗口缩放 (上下文的 5%，上限为 _SUMMARY_TOKENS_CEILING)，
        以便大上下文模型获得更丰富的摘要而不是被硬上限在 8K tokens。
        """
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Serialize conversation turns into labeled text for the summarizer.
        将对话轮次序列化为带标签的文本供摘要器使用。

        包含工具调用参数和结果内容 (每条消息最多 3000 字符)，
        以便摘要器保留特定细节如文件路径、命令和输出。
        """
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a structured summary of conversation turns.
        生成对话轮次的结构化摘要。

        使用结构化模板 (Goal, Progress, Decisions, Files, Next Steps)。
        当存在前一个摘要时，生成迭代更新而非从头总结。

        真正的摘要算法在 ContextCompressor.compress()。
        多次压缩时迭代更新摘要而不是每次重头总结。

        如果所有尝试都失败则返回 None — 调用者应丢弃中间轮次而非注入无用的占位符。
        """
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - now,
            )
            return None

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)

        if self._previous_summary:
            # 迭代更新: 保留现有信息，添加新进度
            prompt = f"""You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new progress. Move items from "In Progress" to "Done" when completed. Remove information only if it is clearly obsolete.

## Goal
[What the user is trying to accomplish — preserve from previous summary, update if goal evolved]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions — accumulate across compactions]

## Progress
### Done
[Completed work — include specific file paths, commands run, results obtained]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why they were made]

## Relevant Files
[Files read, modified, or created — with brief note on each. Accumulate across compactions.]

## Next Steps
[What needs to happen next to continue the work]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation]

Target ~{summary_budget} tokens. Be specific — include file paths, command outputs, error messages, and concrete values rather than vague descriptions.

Write only the summary body. Do not include any preamble or prefix."""
        else:
            # 首次压缩: 从头总结
            prompt = f"""Create a structured handoff summary for a later assistant that will continue this conversation after earlier turns are compacted.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work — include specific file paths, commands run, results obtained]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why they were made]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next to continue the work]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation]

Target ~{summary_budget} tokens. Be specific — include file paths, command outputs, error messages, and concrete values rather than vague descriptions. The goal is to prevent the next assistant from repeating work or losing important details.

Write only the summary body. Do not include any preamble or prefix."""

        try:
            call_kwargs = {
                "task": "compression",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": summary_budget * 2,
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            response = call_llm(**call_kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                content = str(content) if content else ""
            summary = content.strip()
            # 存储以供下次压缩时迭代更新
            self._previous_summary = summary
            self._summary_failure_cooldown_until = 0.0
            return self._with_summary_prefix(summary)
        except RuntimeError:
            self._summary_failure_cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            logging.warning("Context compression: no provider available for "
                            "summary. Middle turns will be dropped without summary "
                            "for %d seconds.",
                            _SUMMARY_FAILURE_COOLDOWN_SECONDS)
            return None
        except Exception as e:
            self._summary_failure_cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            logging.warning(
                "Failed to generate context summary: %s. "
                "Further summary attempts paused for %d seconds.",
                e,
                _SUMMARY_FAILURE_COOLDOWN_SECONDS,
            )
            return None

    @staticmethod
    def _with_summary_prefix(summary: str) -> str:
        """Normalize summary text to the current compaction handoff format.
        将摘要文本规范化为当前压缩交接格式。
        """
        text = (summary or "").strip()
        for prefix in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    # ------------------------------------------------------------------
    # 工具调用/结果对完整性辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression.
        压缩后修复孤立的 tool_call / tool_result 对。

        两种失败模式:
        1. 工具结果引用的 call_id 的 assistant tool_call 被移除 (摘要/截断)。
           API 会拒绝并报错 "No tool call found for function call output with call_id ..."。
        2. assistant 消息有 tool_calls 但其结果被丢弃。
           API 会拒绝因为每个 tool_call 必须后跟带有匹配 call_id 的工具结果。

        此方法移除孤立结果并为孤立调用插入存根结果，
        以使消息列表始终格式良好。
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = self._get_tool_call_id(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self.quiet_mode:
                logger.info("Compression sanitizer: added %d stub tool result(s)", len(missing_results))

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.
        将压缩起始边界向前推过任何孤立工具结果。

        如果 messages[idx] 是工具结果，向前滑动直到遇到非工具消息，
        这样不会在组中间开始摘要区域。
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a tool_call/result group.
        将压缩结束边界向后拉以避免拆分 tool_call/result 组。

        如果边界落在工具结果组中间 (即 idx 前面有连续的工具消息)，
        向后遍历找到父 assistant 消息。如果找到，将边界移到她之前，
        以便整个 assistant + tool_results 组被包含在摘要区域而非被拆分
        (这会导致 _sanitize_tool_pairs 删除孤立尾结果时静默数据丢失)。
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # 按 token 预算的尾部保护
    # ------------------------------------------------------------------

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """Walk backward from the end of messages, accumulating tokens until the budget is reached.
        从消息末尾向后走，累积 token 直到达到预算。返回尾部开始的索引。

        token_budget 默认为 self.tail_token_budget，它源自
        summary_target_ratio * context_length，因此随模型上下文窗口自动缩放。

        不会在 tool_call/result 组内切割。如果预算保护的消息少于 protect_last_n，
        则回退到旧的 protect_last_n 方法。
        """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        min_tail = self.protect_last_n
        accumulated = 0
        cut_idx = n

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            content = msg.get("content") or ""
            msg_tokens = len(content) // _CHARS_PER_TOKEN + 10
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            if accumulated + msg_tokens > token_budget and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        fallback_cut = n - min_tail
        if cut_idx > fallback_cut:
            cut_idx = fallback_cut

        if cut_idx <= head_end:
            cut_idx = fallback_cut

        cut_idx = self._align_boundary_backward(messages, cut_idx)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # 主要压缩入口点
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.
        通过总结中间轮次来压缩对话消息。

        真正的摘要算法在 ContextCompressor.compress()。

        算法:
          1. 修剪旧工具结果 (廉价预传递，无需 LLM 调用)
          2. 保护头部消息 (系统提示 + 首次交换)
          3. 按 token 预算找到尾部边界 (最近约 20K tokens 的上下文)
          4. 用结构化 LLM 提示总结中间轮次
          5. 重新压缩时，迭代更新前一个摘要

        压缩后，清理孤立的 tool_call / tool_result 对，
        以使 API 永远不会收到不匹配的 ID。
        """
        # 【文档锚点 4A】上下文压缩本体：保头、保尾、摘要化中间段，生成可交接的新上下文
        n_messages = len(messages)
        if n_messages <= self.protect_first_n + self.protect_last_n + 1:
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need > %d)",
                    n_messages,
                    self.protect_first_n + self.protect_last_n + 1,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: 修剪旧工具结果 (廉价，无需 LLM 调用)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n * 3,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: 确定边界
        compress_start = self.protect_first_n
        compress_start = self._align_boundary_forward(messages, compress_start)

        # 使用 token 预算尾部保护而非固定消息数
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # Phase 3: 生成结构化摘要
        summary = self._generate_summary(turns_to_summarize)

        # Phase 4: 组装压缩后的消息列表
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = (
                    (msg.get("content") or "")
                    + "\n\n[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work.]"
                )
            compressed.append(msg)

        _merge_summary_into_tail = False
        if summary:
            last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
            first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
            if last_head_role in ("assistant", "tool"):
                summary_role = "user"
            else:
                summary_role = "assistant"
            if summary_role == first_tail_role:
                flipped = "assistant" if summary_role == "user" else "user"
                if flipped != last_head_role:
                    summary_role = flipped
                else:
                    _merge_summary_into_tail = True
            if not _merge_summary_into_tail:
                compressed.append({"role": summary_role, "content": summary})
        else:
            if not self.quiet_mode:
                logger.debug("No summary model available — middle turns dropped without summary")

        for i in range(compress_end, n_messages):
            msg = messages[i].copy()
            if _merge_summary_into_tail and i == compress_end:
                original = msg.get("content") or ""
                msg["content"] = summary + "\n\n" + original
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        if not self.quiet_mode:
            new_estimate = estimate_messages_tokens_rough(compressed)
            saved_estimate = display_tokens - new_estimate
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved)",
                n_messages,
                len(compressed),
                saved_estimate,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed
