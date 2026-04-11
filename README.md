# Hello Hermes Agent ☤

This is a workspace for exploring and analyzing [Hermes Agent](https://github.com/nousresearch/hermes-agent) `v0.8.0 (v2026.4.8)` `86960cdb`.

> ## Pronunciation

> Note: The "Hermes" in this project refers to the Greek deity.
> 
> ✔️ **Hermes**: `/ˈhɜːrmiːz/` — The Greek god of language and writing, and messenger of the gods.
> 
> ✖️ **Hermès**: `/ɛʁ.mɛs/` — French luxury brand.

## Resources

- **Official Repository**: https://github.com/nousresearch/hermes-agent
- **Official Website**: https://hermes-agent.nousresearch.com
- **Quickstart Documentation**: https://hermes-agent.nousresearch.com/docs/getting-started/quickstart

## Quick Start

```bash
# Run the setup wizard
hermes setup

# View/edit configuration
code ~/.hermes/

# Start interactive chat
hermes
```

## 源代码分析

- [hermes-agent v2026.4.8 源代码分析](hermes-agent v2026.4.8 源代码分析.md)

### 1. 核心架构全览 (v2026.4.8)

Hermes 不仅是一个 CLI，而是一个高度分层的 Agent 编排系统：

| 层级 | 主要模块 | 核心职责 |
| --- | --- | --- |
| **入口层** | `hermes`, `pyproject.toml` | 暴露命令、选择运行面（CLI/Gateway/ACP） |
| **控制层** | `hermes_cli/main.py` | 处理 Profile、环境变量、日志、子命令分发 |
| **外壳层** | `cli.py`, `gateway/run.py` | 装配终端或平台运行时，把输入整理成统一会话 |
| **内核层** | `run_agent.py:AIAgent` | **核心编排器**：构建提示词、模型循环、执行工具、压缩、持久化 |
| **能力层** | `agent/`, `tools/`, `model_tools.py` | 提供工具自发现、外部记忆编排、结构化上下文压缩 |
| **持久化层** | `hermes_state.py`, `gateway/session.py` | 维护 SQLite (SessionDB) 与 JSONL 消息流的同步 |

### 2. 启动链与代码映射：`hermes` -> `AIAgent`

以下链路展示了从终端命令到内核实例化的精确跳转逻辑及**调用点 (Call Site)**：

```mermaid
%%{init: {'flowchart': {'curve': 'basis'}, 'theme': 'neutral'}}%%
flowchart TD
    A["hermes / launcher<br/><i>hermes:1-10</i>"] 
    B["hermes_cli.main:main<br/><i>main.py:4127-4408</i>"]
    C["hermes_cli.main:cmd_chat<br/><i>main.py:556-663</i>"]
    D["cli:main<br/><i>cli.py:8525-8732</i>"]
    E["HermesCLI:__init__<br/><i>cli.py:1315-1580</i>"]
    F["HermesCLI:_init_agent<br/><i>cli.py:2363-2485</i>"]
    G["AIAgent:__init__<br/><i>run_agent.py:433-1215</i>"]

    A -->|import & call| B
    B -->|parser.set_defaults @ L4308| C
    C -->|cli_main(**kwargs) @ L662| D
    D -->|cli = HermesCLI(...) @ L8655| E
    D -.->|if cli._init_agent() @ L8698| F
    E -->|via cli.run @ L8732 -> chat| F
    F -->|self.agent = AIAgent(...) @ L2423| G

    subgraph CoreComponents ["内核组件装配 (run_agent.py)"]
        G1["Provider/Client Setup<br/><i>line 553-829</i>"]
        G2["Tool Discovery<br/><i>model_tools.py:234-353</i>"]
        G3["Memory & Compressor<br/><i>line 969-1145</i>"]
    end
    
    G --> G1 & G2 & G3

    %% 颜色分类
    classDef start fill:#f6f8fa,stroke:#24292e,stroke-width:1px;
    classDef process fill:#f5faff,stroke:#005cc5,stroke-width:1px;
    classDef core fill:#fff5eb,stroke:#e36209,stroke-width:1px;
    classDef components fill:#f0fff4,stroke:#22863a,stroke-width:1px,stroke-dasharray: 5 5;

    class A start;
    class B,C,D,E,F process;
    class G core;
    class G1,G2,G3 components;
```

#### 关键跳转点全解析：
1.  **统一入口分发**：`hermes` 脚本在 `L10` 导入并调用 `hermes_cli.main:main`。
2.  **命令解析与路由**：`main.py` 通过 `argparse` 的 `set_defaults(func=cmd_chat)` 在 `L4308` 完成子命令绑定；解析完成后通过 `args.func(args)` 进入 `cmd_chat`。
3.  **外壳环境切换**：`cmd_chat` 在完成会话 ID 解析和环境变量设置后，于 **`main.py:662`** 调用 `cli_main(**kwargs)`，正式进入 `cli.py`。
4.  **外壳实例化**：`cli.py` 的 `main` 函数在 **`L8655`** 通过 `cli = HermesCLI(...)` 触发 `__init__`。此时仅做配置加载与 UI 初始化。
5.  **内核延迟加载**：
    -   **单次查询模式**：在 `cli.py:8698` 直接调用 `_init_agent()`。
    -   **交互模式**：`main` 调用 `cli.run()` (`L8732`)，随后通过 `chat()` 循环触发 `_init_agent()`。
6.  **进入内核**：`HermesCLI._init_agent` 在完成凭据校验后，于 **`cli.py:2423`** 实例化 `AIAgent`。
7.  **内核能力装配**：`AIAgent.__init__` 启动时，同步完成 Provider Client 创建 (`L553`)、工具自发现 (`model_tools.py:234`) 以及记忆管理器初始化 (`L969`)。


### 3. 核心进化机制

#### 3.1 自我进化 (Self-Improvement)
Hermes 在每个对话 Turn 结束后并不会立即停止，而是通过以下三层机制实现自我更新：
1.  **记忆沉淀**：在压缩发生前调用 `flush_memories()`，强制模型提取长期偏好写入 `MEMORY.md`。
2.  **技能沉淀**：通过 `skill_manage` 工具，将复杂的 SOP 沉淀为可重用的 `.py` 脚本存入 `skills/` 目录。
3.  **后台复盘 (Background Review)**：响应用户后，Agent 会 Fork 一个静默实例，基于本轮对话进行 `_spawn_background_review()`，判断是否需要更新记忆或补丁现有技能。

#### 3.2 Turn 级动态路由 (Smart Routing)
通过 `_resolve_turn_agent_config()` (`cli.py:2344`)，Agent 会在每一轮对话前动态评估输入复杂度。如果是简单问候或短文本，会切到廉价模型（Cheap Route）；如果是复杂任务，则回落到主模型（Primary Route）。这种模型路由是 **Turn 级** 的，而非进程级固定的。

----


<img src="hello-hermes.png" alt="hello-hermes" style="height:800px; display: block; margin-left: 0;" />