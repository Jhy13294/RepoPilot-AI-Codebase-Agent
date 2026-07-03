# RepoPilot 🛩️

**面向代码仓库的任务型 Agent：Issue 分析与 Patch 建议，全程人工审批保护。**

[English → README.md](README.md)

RepoPilot **不是聊天机器人**。给它一个仓库和一个 Issue，它会：制定计划 → 用类型化工具阅读代码 →
定位根因 → 以可审查的 diff 形式提出修复 → **等待你批准后**才落盘 → 运行测试 → 失败时自动恢复重试 →
输出修复报告和完整的机器可读执行轨迹。

> 状态：**Phase 0 — 文档先行的骨架阶段。** 架构与契约已定稿，功能按[路线图](docs/roadmap.md)逐阶段落地。

## 项目定位

一个聚焦 LLM Agent 工程化难点的作品集项目：

- **Agent 编排** — 基于原生 tool calling 手写 Planner–Executor–Critic 状态机（[为什么不用 LangGraph](docs/tech-selection.md)）
- **规范的工具调用** — 每个工具都有 Pydantic 入/出参 schema、风险等级、超时与截断上限、统一结果信封
- **Human-in-the-loop 安全** — 高风险动作（写文件、打补丁、跑测试、git 操作）由**代码层**的审批门拦截，绝不信任模型自我约束
- **失败恢复** — 类型化错误路由到类型化策略（参数修复、重读重生成、修复循环），全部受预算约束
- **可观测与可评测** — JSONL 轨迹驱动指标评测（成功率、工具调用准确率、恢复率、成本）

## 能力规划（按阶段交付）

| 能力 | 阶段 |
|---|---|
| 带真实 file:line 引用的仓库问答 | P2 |
| Issue 分析 → 嫌疑文件 + 根因 + 置信度 | P4 |
| 生成 unified diff，**审批后**才应用 | P5 |
| 运行测试、失败恢复、预算内重新修复 | P6 |
| 任务集指标评测报告 | P7 |
| Web 控制台：实时轨迹 + 审批面板 | P8 |

## 架构一览

完整组件图与时序图见 [docs/architecture.md](docs/architecture.md)。核心链路：

用户 → 编排器（Planner / Executor / Critic）→ 工具注册中心 → **审批门** → 路径沙箱 → 工作区仓库，
全过程写入 Trace（JSONL + SQLite）。

## 快速开始（Phase 2 起可用）

```bash
git clone <this-repo> && cd RepoPilot
uv sync
cp .env.example .env          # 填入 DeepSeek API Key（默认 provider，模型 deepseek-v4-pro）
uv run repopilot ask ./path/to/repo "日期解析在哪里处理？"
uv run repopilot run ./path/to/repo --issue "配置文件为空时抛 TypeError"
```

## 安全模型（简版）

1. 每个工具声明 `risk_level: low | medium | high`；
2. `high`（apply_patch、run_tests、git 变更）**阻塞等待人工批准** —— 你会看到真实 diff 和 Agent 的理由；
3. 所有路径经沙箱校验；补丁只落在 `repopilot/fix-*` 分支，绝不动你的分支；
4. 刻意**不提供 `run_shell`** 工具。

详见 [docs/human-in-the-loop.md](docs/human-in-the-loop.md)。

## 📚 学习笔记 / Learning Notes

> 文档是一等交付物：设计取舍、失败教训，都写成可读的东西。

| 文档 | 内容 |
|---|---|
| [技术选型](docs/tech-selection.md) | 每项选择的理由 + LangGraph 取舍 |
| [架构设计](docs/architecture.md) | 组件、数据流、关键接口、图 |
| [Agent 设计](docs/agent-design.md) | 状态机、预算、Prompt 架构、上下文管理 |
| [工具调用设计](docs/tool-calling-design.md) | 注册中心模式 + 每个工具的完整规格 |
| [人工审批](docs/human-in-the-loop.md) | 风险分级、审批流、不可绕过性测试 |
| [失败恢复](docs/failure-recovery.md) | 错误分类 → 恢复策略、反模式 |
| [评测设计](docs/evaluation.md) | 任务类型、指标、评测框架、结果记录 |
| [路线图](docs/roadmap.md) | Phase 0–9 与验收标准 |
| [项目管理](docs/project-management.md) | 施工编号、任务生命周期、完成定义 |
| [代码规范](docs/code-style.md) | 语言政策、工具链、类型、提交规范 |

## 工程方法

用施工编号（如 `RP-P1-FEAT-003`）串联任务看板、分支名与提交信息。内部看板（`tasks/`、`memory/`）
不入库，公开模板见 [docs/internal-templates/](docs/internal-templates/)，方法论见
[docs/project-management.md](docs/project-management.md)。

## 技术栈

Python 3.12 · uv · FastAPI · Pydantic v2 · DeepSeek `deepseek-v4-pro`（OpenAI 兼容 SDK，+ Anthropic 适配器）·
SQLAlchemy 2.0（SQLite → PostgreSQL）· Streamlit · pytest · ruff · mypy · Docker

## 许可证

[MIT](LICENSE)
