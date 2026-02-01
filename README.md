# Auto-Quant Agent 模板（Cursor 版）

这是一个可持续运行的 Auto-Quant Agent 模板，用于让 AI 持续优化股票/ETF 策略参数：

- **AI 只改参数文件**：`strategy/strategy_params.py`
- **回测固定不动**：`backtest/` 负责生成结构化指标 `metrics.json`
- **Agent 闭环**：`agent/agent.py` 负责跑回测 → 评分 → 生成提示词 → 等待下一轮
- **护栏**：防止参数 key 变更、参数越界、或出现可疑模式；违规自动回滚
- **可复现**：每轮迭代都会落盘到 `runs/iter_xxxx/`

> 适用场景：参数调优、因子窗口/权重微调、风控阈值调优。该模板刻意避免让 AI 直接改策略结构，以降低“失控/过拟合/作弊”的风险。

---

## 1. 目录结构

```text
.
├── agent/
│   ├── agent.py              # 主循环：回测→评估→生成prompt→等待
│   ├── evaluator.py          # 多目标评分函数
│   ├── guardrails.py         # 护栏：禁止模式/参数范围/key不变
│   ├── llm_client.py         # LLM 接口（当前为 Cursor 手动模式占位）
│   └── prompt_builder.py     # 组装给 AI 的提示词
│
├── strategy/
│   ├── base_strategy.py      # 冻结策略结构（不改）
│   ├── epo_strategy.py       # 示例策略结构（不改）
│   ├── features.py           # 因子/工具（不改）
│   └── strategy_params.py    # ✅ AI 只允许改这里
│
├── backtest/
│   ├── engine.py             # 回测引擎
│   ├── metrics.py            # 指标计算
│   └── run_backtest.py       # 回测入口，输出 metrics.json
│
├── data/
│   └── processed/
│       └── prices.csv        # （可选）你的真实数据：至少包含 close 列
│
├── runs/
│   └── iter_0000/ ...         # 每轮迭代归档（自动生成）
│
├── config.yaml               # 迭代轮数/目标/权重/护栏/回测命令
├── metrics.json              # 最近一次回测输出（自动生成）
├── agent_prompt.txt          # 最近一次提示词（自动生成）
├── requirements.txt
└── README.md
```

---

## 2. 安装与运行

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 跑一次回测（先验证链路）

```bash
python -m backtest.run_backtest
```

成功后会在根目录生成：

- `metrics.json`

### 2.3 启动 Agent 闭环（可跑一夜）

```bash
python -m agent.agent
```

Agent 每一轮会：

- 运行 `config.yaml` 里的 `backtest_cmd`
- 读取 `metrics.json`
- 计算 score
- 写出 prompt：
  - `agent_prompt.txt`（根目录，方便 Cursor 直接看）
  - `runs/iter_xxxx/agent_prompt.txt`（归档）
- 等待 `sleep_seconds` 秒进入下一轮

---

## 3. 如何用 Cursor 让 AI 持续优化

### 3.1 你需要打开的文件

- `agent_prompt.txt`
- `strategy/strategy_params.py`

### 3.2 给 Cursor AI 的“铁律指令”（强烈建议复制粘贴）

你可以在 Cursor 对 AI 说：

- **只根据 `agent_prompt.txt` 的反馈修改 `strategy/strategy_params.py`。**
- **不要修改其他任何文件。**
- **不得新增/删除任何参数 key。**
- **输出 `strategy/strategy_params.py` 的完整文件内容（只输出代码，不要解释）。**

### 3.3 运行节奏

- Agent 生成 `agent_prompt.txt`
- 你让 Cursor AI 改 `strategy/strategy_params.py` 并保存
- Agent 到点进入下一轮，自动跑回测

如果你想给 AI 更多反应时间：

- 修改 `config.yaml`：`agent.sleep_seconds`

---

## 4. 数据接入（用你自己的 ETF/股票数据）

回测入口 `backtest/run_backtest.py` 的数据加载规则：

- **优先读取**：`data/processed/prices.csv`
  - 要求：至少包含 `close` 列
  - 可选：`date` 列（如果有，会被作为时间索引）
- 如果没有该文件，会用内置随机数据生成一段价格序列（用于确保模板开箱即跑）

建议你把真实标的/策略的数据预处理到 `data/processed/prices.csv`，并保证 `close` 列表示收盘价。

---

## 5. 评分、目标与护栏（防失控/防作弊）

### 5.1 目标与权重

在 `config.yaml`：

- `objectives.target`：目标阈值（用于提示词）
- `objectives.weights`：评分权重

当前默认评分：

```text
score = 0.4*sharpe + 0.3*annual_return - 0.3*max_drawdown - 0.1*turnover
```

你可以按你的偏好调整权重。

### 5.2 护栏规则

在 `config.yaml`：

- `constraints.forbid_new_keys: true`
  - 禁止增删改 `PARAMS` 的 key
- `constraints.params_range`
  - 参数范围限制
- `constraints.forbidden_patterns`
  - 禁止出现的文本模式（例如可疑未来函数/作弊迹象）

违规后 Agent 会：

- 将 `strategy/strategy_params.py` 回滚到本轮开始前版本
- 在 `runs/iter_xxxx/guardrails_error.txt` 记录原因

---

## 6. 每轮产物（runs/iter_xxxx）解释

每一轮会保存：

- `params_before.py`：本轮开始前参数
- `params_after.py`：本轮结束后的参数（可能被回滚）
- `metrics.json`：本轮指标
- `score.txt`：本轮得分
- `agent_prompt.txt`：本轮给 AI 的提示词
- `summary.json`：本轮摘要（含 best_score/best_iter）

这保证：

- **可复现**：任何一次结果都能对应到当时的参数
- **可回滚**：你可以手动把某轮 `params_after.py` 复制回 `strategy/strategy_params.py`

---

## 7. 以后怎么扩展（推荐路线）

- **加入 OOS / Walk-forward**：把数据切分为 train/valid/test，评分只用 valid/test
- **多标的/多资产**：`prices.csv` 扩展为多列 close（或多文件）并在策略/回测中支持
- **真正无人值守**：在 `agent/llm_client.py` 增加 OpenAI/DeepSeek/Qwen 等 API 调用，让 Agent 自动写回 `strategy_params.py`
- **更强护栏**：
  - 静态扫描（AST）限制只能出现 `PARAMS = {...}`
  - 限制参数改动幅度（比如每轮不超过 ±20%）
  - 失败重试/超时

---

## 8. 最常见的使用姿势（总结）

- 你只改/只让 AI 改：`strategy/strategy_params.py`
- 你只需要运行：`python -m agent.agent`
- 你只需要盯：`agent_prompt.txt` + `runs/`
