#!/usr/bin/env python3
"""工具集模块 - 工具分组与动态解析

提供灵活的工具别名/工具集定义和管理系统。工具集允许将工具分组以适应特定场景，
可由单个工具或其他工具集组成。

核心数据结构：
- _HERMES_CORE_TOOLS: 所有平台的共享工具列表
- TOOLSETS: 工具集定义字典（description + tools + includes）
"""

from typing import List, Dict, Any, Set, Optional


# CLI 和所有消息平台工具集共享的工具列表
_HERMES_CORE_TOOLS = [
    # Web
    "web_search", "web_extract",
    # 终端 + 进程管理
    "terminal", "process",
    # 文件操作
    "read_file", "write_file", "patch", "search_files",
    # 视觉 + 图像生成
    "vision_analyze", "image_generate",
    # 技能
    "skills_list", "skill_view", "skill_manage",
    # 浏览器自动化
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images",
    "browser_vision", "browser_console",
    # 文本转语音
    "text_to_speech",
    # 规划与记忆
    "todo", "memory",
    # 会话历史搜索
    "session_search",
    # 澄清问题
    "clarify",
    # 代码执行 + 委托
    "execute_code", "delegate_task",
    # Cronjob 管理
    "cronjob",
    # 跨平台消息发送（通过 check_fn 限制 gateway 运行时）
    "send_message",
    # Home Assistant 智能家居控制（通过 check_fn 限制 HASS_TOKEN）
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
]


# 工具集定义
TOOLSETS = {
    # 基础工具集 - 单个工具类别
    "web": {
        "description": "Web research and content extraction tools",
        "tools": ["web_search", "web_extract"],
        "includes": []
    },

    "search": {
        "description": "Web search only (no content extraction/scraping)",
        "tools": ["web_search"],
        "includes": []
    },

    "vision": {
        "description": "Image analysis and vision tools",
        "tools": ["vision_analyze"],
        "includes": []
    },

    "image_gen": {
        "description": "Creative generation tools (images)",
        "tools": ["image_generate"],
        "includes": []
    },

    "terminal": {
        "description": "Terminal/command execution and process management tools",
        "tools": ["terminal", "process"],
        "includes": []
    },

    "moa": {
        "description": "Advanced reasoning and problem-solving tools",
        "tools": ["mixture_of_agents"],
        "includes": []
    },

    "skills": {
        "description": "Access, create, edit, and manage skill documents with specialized instructions and knowledge",
        "tools": ["skills_list", "skill_view", "skill_manage"],
        "includes": []
    },

    "browser": {
        "description": "Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs",
        "tools": [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console", "web_search"
        ],
        "includes": []
    },

    "cronjob": {
        "description": "Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks",
        "tools": ["cronjob"],
        "includes": []
    },

    "messaging": {
        "description": "Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc.",
        "tools": ["send_message"],
        "includes": []
    },

    "rl": {
        "description": "RL training tools for running reinforcement learning on Tinker-Atropos",
        "tools": [
            "rl_list_environments", "rl_select_environment",
            "rl_get_current_config", "rl_edit_config",
            "rl_start_training", "rl_check_status",
            "rl_stop_training", "rl_get_results",
            "rl_list_runs", "rl_test_inference"
        ],
        "includes": []
    },

    "file": {
        "description": "File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files)",
        "tools": ["read_file", "write_file", "patch", "search_files"],
        "includes": []
    },

    "tts": {
        "description": "Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, or OpenAI",
        "tools": ["text_to_speech"],
        "includes": []
    },

    "todo": {
        "description": "Task planning and tracking for multi-step work",
        "tools": ["todo"],
        "includes": []
    },

    "memory": {
        "description": "Persistent memory across sessions (personal notes + user profile)",
        "tools": ["memory"],
        "includes": []
    },

    "session_search": {
        "description": "Search and recall past conversations with summarization",
        "tools": ["session_search"],
        "includes": []
    },

    "clarify": {
        "description": "Ask the user clarifying questions (multiple-choice or open-ended)",
        "tools": ["clarify"],
        "includes": []
    },

    "code_execution": {
        "description": "Run Python scripts that call tools programmatically (reduces LLM round trips)",
        "tools": ["execute_code"],
        "includes": []
    },

    "delegation": {
        "description": "Spawn subagents with isolated context for complex subtasks",
        "tools": ["delegate_task"],
        "includes": []
    },

    "homeassistant": {
        "description": "Home Assistant smart home control and monitoring",
        "tools": ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"],
        "includes": []
    },

    # 场景特定工具集

    "debugging": {
        "description": "Debugging and troubleshooting toolkit",
        "tools": ["terminal", "process"],
        "includes": ["web", "file"]
    },

    "safe": {
        "description": "Safe toolkit without terminal access",
        "tools": [],
        "includes": ["web", "vision", "image_gen"]
    },

    # 全 Hermes 工具集（CLI + 消息平台）
    "hermes-acp": {
        "description": "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI",
        "tools": [
            "web_search", "web_extract",
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze",
            "skills_list", "skill_view", "skill_manage",
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console",
            "todo", "memory",
            "session_search",
            "execute_code", "delegate_task",
        ],
        "includes": []
    },

    "hermes-api-server": {
        "description": "OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message)",
        "tools": [
            "web_search", "web_extract",
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "skills_list", "skill_view", "skill_manage",
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_get_images",
            "browser_vision", "browser_console",
            "todo", "memory",
            "session_search",
            "execute_code", "delegate_task",
            "cronjob",
            "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
        ],
        "includes": []
    },

    "hermes-cli": {
        "description": "Full interactive CLI toolset - all default tools plus cronjob management",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-telegram": {
        "description": "Telegram bot toolset - full access for personal use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-discord": {
        "description": "Discord bot toolset - full access (terminal has safety checks via dangerous command approval)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-whatsapp": {
        "description": "WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-slack": {
        "description": "Slack bot toolset - full access for workspace use (terminal has safety checks)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-signal": {
        "description": "Signal bot toolset - encrypted messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-homeassistant": {
        "description": "Home Assistant bot toolset - smart home event monitoring and control",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-email": {
        "description": "Email bot toolset - interact with Hermes via email (IMAP/SMTP)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-mattermost": {
        "description": "Mattermost bot toolset - self-hosted team messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-matrix": {
        "description": "Matrix bot toolset - decentralized encrypted messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-dingtalk": {
        "description": "DingTalk bot toolset - enterprise messaging platform (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-feishu": {
        "description": "Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-wecom": {
        "description": "WeCom bot toolset - enterprise WeChat messaging (full access)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-sms": {
        "description": "SMS bot toolset - interact with Hermes via SMS (Twilio)",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-webhook": {
        "description": "Webhook toolset - receive and process external webhook events",
        "tools": _HERMES_CORE_TOOLS,
        "includes": []
    },

    "hermes-gateway": {
        "description": "Gateway toolset - union of all messaging platform tools",
        "tools": [],
        "includes": [
            "hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack",
            "hermes-signal", "hermes-homeassistant", "hermes-email", "hermes-sms",
            "hermes-mattermost", "hermes-matrix", "hermes-dingtalk", "hermes-feishu",
            "hermes-wecom", "hermes-webhook"
        ]
    }
}


def get_toolset(name: str) -> Optional[Dict[str, Any]]:
    """按名称获取工具集定义"""
    return TOOLSETS.get(name)


def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    """递归解析工具集获取所有工具名称

    处理工具集组合：递归解析包含的工具集并合并所有工具。
    支持别名 "all" 和 "*" 表示所有工具。
    自动检测循环引用（菱形依赖）并跳过。
    """
    if visited is None:
        visited = set()

    # 别名：all 或 * 代表所有工具
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return list(all_tools)

    # 检测循环/已解析
    if name in visited:
        return []

    visited.add(name)

    toolset = TOOLSETS.get(name)
    if not toolset:
        # 回退到工具注册中心获取插件提供的工具集
        if name in _get_plugin_toolset_names():
            try:
                from tools.registry import registry
                return [e.name for e in registry._tools.values() if e.toolset == name]
            except Exception:
                pass
        return []

    # 收集直接工具
    tools = set(toolset.get("tools", []))

    # 递归解析包含的工具集
    for included_name in toolset.get("includes", []):
        included_tools = resolve_toolset(included_name, visited)
        tools.update(included_tools)

    return list(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """解析多个工具集并合并其工具"""
    all_tools = set()

    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)

    return list(all_tools)


def _get_plugin_toolset_names() -> Set[str]:
    """返回插件注册的工具集名称（来自工具注册中心但不在 TOOLSETS 字典中）"""
    try:
        from tools.registry import registry
        return {
            entry.toolset
            for entry in registry._tools.values()
            if entry.toolset not in TOOLSETS
        }
    except Exception:
        return set()


def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """获取所有可用工具集及其定义

    包含静态定义的工具集和插件注册的工具集。
    """
    result = TOOLSETS.copy()
    # 添加插件提供的工具集
    for ts_name in _get_plugin_toolset_names():
        if ts_name not in result:
            try:
                from tools.registry import registry
                tools = [e.name for e in registry._tools.values() if e.toolset == ts_name]
                result[ts_name] = {
                    "description": f"Plugin toolset: {ts_name}",
                    "tools": tools,
                }
            except Exception:
                pass
    return result


def get_toolset_names() -> List[str]:
    """获取所有可用工具集名称（不包括别名）

    包含插件注册的工具集名称。
    """
    names = set(TOOLSETS.keys())
    names |= _get_plugin_toolset_names()
    return sorted(names)


def validate_toolset(name: str) -> bool:
    """检查工具集名称是否有效"""
    if name in {"all", "*"}:
        return True
    if name in TOOLSETS:
        return True
    return name in _get_plugin_toolset_names()


def create_custom_toolset(
    name: str,
    description: str,
    tools: List[str] = None,
    includes: List[str] = None
) -> None:
    """运行时创建自定义工具集"""
    TOOLSETS[name] = {
        "description": description,
        "tools": tools or [],
        "includes": includes or []
    }


def get_toolset_info(name: str) -> Dict[str, Any]:
    """获取工具集的详细信息（包含解析后的工具）"""
    toolset = get_toolset(name)
    if not toolset:
        return None

    resolved_tools = resolve_toolset(name)

    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved_tools,
        "tool_count": len(resolved_tools),
        "is_composite": bool(toolset["includes"])
    }


if __name__ == "__main__":
    print("Toolsets System Demo")
    print("=" * 60)

    print("\nAvailable Toolsets:")
    print("-" * 40)
    for name, toolset in get_all_toolsets().items():
        info = get_toolset_info(name)
        composite = "[composite]" if info["is_composite"] else "[leaf]"
        print(f"  {composite} {name:20} - {toolset['description']}")
        print(f"     Tools: {len(info['resolved_tools'])} total")

    print("\nToolset Resolution Examples:")
    print("-" * 40)
    for name in ["web", "terminal", "safe", "debugging"]:
        tools = resolve_toolset(name)
        print(f"\n  {name}:")
        print(f"    Resolved to {len(tools)} tools: {', '.join(sorted(tools))}")

    print("\nMultiple Toolset Resolution:")
    print("-" * 40)
    combined = resolve_multiple_toolsets(["web", "vision", "terminal"])
    print("  Combining ['web', 'vision', 'terminal']:")
    print(f"    Result: {', '.join(sorted(combined))}")

    print("\nCustom Toolset Creation:")
    print("-" * 40)
    create_custom_toolset(
        name="my_custom",
        description="My custom toolset for specific tasks",
        tools=["web_search"],
        includes=["terminal", "vision"]
    )
    custom_info = get_toolset_info("my_custom")
    print("  Created 'my_custom' toolset:")
    print(f"    Description: {custom_info['description']}")
    print(f"    Resolved tools: {', '.join(custom_info['resolved_tools'])}")
