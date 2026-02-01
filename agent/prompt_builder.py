"""agent/prompt_builder.py

Prompt 构建与落盘。

该模块负责把“当前回测结果 + 综合得分 + 优化目标 + 约束条件 + 当前参数文件内容”
拼接成一段明确可执行的指令文本（prompt），用于驱动人工或 LLM 去修改策略参数文件。

设计要点：
- prompt 里明确要求输出 `strategy/strategy_params.py` 的完整文件内容，方便直接覆盖写入。
- 约束部分会根据 config 中的 `constraints` 动态生成（例如是否禁止新增 key、参数范围）。
- `base_params_text` 会原样嵌入代码块，确保修改时有上下文。
"""

from pathlib import Path


def build_prompt(*, metrics: dict, score: float, target: dict, constraints: dict, base_params_text: str) -> str:
    """构建用于指导修改策略参数文件的 prompt 文本。

    参数:
        metrics: 回测输出指标 dict。
        score: 本轮综合评分（由 evaluator 计算）。
        target: 优化目标阈值 dict（例如 sharpe/max_drawdown）。
        constraints: 约束 dict（例如 params_range/forbid_new_keys 等）。
        base_params_text: 当前参数文件 strategy_params.py 的完整文本。

    返回:
        可直接写入文本文件的 prompt 字符串。
    """

    # 将关键指标展开，便于人类/LLM 快速定位问题。
    return (
        "你是一个量化研究助理。\n\n"
        "【当前指标】\n"
        f"Sharpe: {metrics.get('sharpe')}\n"
        f"MaxDrawdown: {metrics.get('max_drawdown')}\n"
        f"AnnualReturn: {metrics.get('annual_return')}\n"
        f"Turnover: {metrics.get('turnover')}\n"
        f"Score: {score}\n\n"
        "【优化目标】\n"
        f"- Sharpe > {target.get('sharpe')}\n"
        f"- MaxDrawdown < {target.get('max_drawdown')}\n\n"
        "【约束】\n"
        "- 只允许修改 strategy/strategy_params.py\n"
        + ("- 不得新增/删除参数 key\n" if constraints.get("forbid_new_keys", True) else "")
        + f"- 参数范围 ∈ [{constraints.get('params_range', {}).get('min', 0.1)}, {constraints.get('params_range', {}).get('max', 200)}]\n"
        + "- 不得使用未来信息 / 不得引入未来函数\n\n"
        + "请输出 strategy/strategy_params.py 的【完整文件内容】（只输出代码，不要解释）。\n\n"
        + "【当前 strategy_params.py】\n"
        + "```python\n"
        + f"{base_params_text}\n"
        + "```\n"
    )


def write_agent_prompt(path: str | Path, text: str) -> None:
    """将 prompt 写入到指定路径（UTF-8）。"""

    Path(path).write_text(text, encoding="utf-8")
