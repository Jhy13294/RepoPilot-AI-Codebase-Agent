# 学习笔记 · Learning Notes

> 按时间倒序。每条记录：现象 → 修复 → 经验。
> First-person engineering log (in Chinese). Entry format: symptom → fix → lesson.

---

## 2026-07-04 · pip 和 uv 装出来的是两个世界

**现象**：本地 `.venv` 最早是用 pip 建的，装到的版本贴着 pyproject 的下限走：ruff 0.5、
mypy 1.10、pytest 8。后来切到 `uv sync`，同一份 pyproject，解析出来的却是 ruff 0.15、
mypy 2.1、pytest 9——前后差出大半年的版本，而在此之前所有"检查通过"都是在旧版本上跑的。

**修复**：先在干净克隆上 `uv sync` 重跑全套检查（ruff / format / pytest / mypy），确认新版本下
仍然全绿，再把 `uv.lock` 提交进仓库，把环境钉死。

**经验**：`>=` 约束声明的是"能接受什么"，不是"实际用的是什么"；同一份依赖声明，两次安装可以
得到完全不同的环境。lockfile 应该跟第一行代码一起入库，而不是等出问题再补。另外，验证要在
干净克隆里做——本机 `.venv` 的状态不可信。

## 2026-07-04 · env_prefix 会安静地吞掉不带前缀的环境变量

**现象**：配置项大多带 `REPOPILOT_` 前缀，但 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、
`ANTHROPIC_API_KEY` 沿用行业惯例名，不带前缀。pydantic-settings 配了
`env_prefix="REPOPILOT_"` 之后，这三个变量会被直接无视——不报错、不警告。结果就是：用户明明
`export` 了 key，程序却坚持说缺 key。

**修复**：这三个字段单独用 `Field(validation_alias="OPENAI_API_KEY")` 显式绑定完整变量名
（`validation_alias` 不参与前缀拼接），并为别名字段单独写了测试断言。

**经验**：最危险的配置错误是"不报错的那种"。写配置加载器之前，把 `.env` 契约里所有不规则处
（这里是前缀不统一）逐条列出来写进任务说明，比依赖实现时的直觉可靠得多。

## 2026-07-03 · 选型时先翻 issue 区，再看功能列表

**现象**：确定用 deepseek-v4-pro 之前翻了一圈相关 issue，发现一个已知问题
（DeepSeek-V3 #1244）：tool call 偶尔不走结构化的 `tool_calls` 字段，而是以纯文本形式混在
`content` 里返回。官方文档只会告诉你"支持 function calling"。

**修复**（预案，尚未实装）：直接写进设计——adapter 层负责检测 `content` 里的工具调用形态，
重解析或要求模型重发；Phase 2 加一个 contract test 长期盯住这个行为。

**经验**："支持某功能"和"该功能可靠"是两回事。选型阶段花半小时读 issue tracker，比上线后
debug 半天便宜；发现的坑当场变成设计约束，而不是留给未来的自己。
