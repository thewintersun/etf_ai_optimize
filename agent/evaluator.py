"""agent/evaluator.py

指标打分模块。

该模块的职责很单一：
- 接收回测输出的 `metrics`（dict）。
- 按 `weights`（dict）对关键指标做线性加权求和，得到一个标量 score。

注意：
- 这里的 score 只是一个“用于优化迭代的信号”，不一定等价于最终策略好坏。
- 由于 metrics 可能来自 JSON，指标值可能是字符串/数字/缺失；因此做了容错转换。
"""


def score(metrics: dict, weights: dict) -> float:
    """根据 metrics 与 weights 计算综合分数。

    参数:
        metrics: 回测输出指标，例如 sharpe/annual_return/max_drawdown/turnover 等。
        weights: 每个指标的权重（缺失则视为 0）。

    返回:
        线性加权后的 score（float）。

    实现细节:
        - 内部 helper `g()` 用于安全地从 metrics 取值并转换为 float。
        - 当取值失败（类型异常/无法转换）时回落到 default。
    """

    def g(k: str, default: float = 0.0) -> float:
        # metrics 很可能是从 JSON 读出来的，value 可能是 str/int/float/None。
        v = metrics.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    # 采用线性模型聚合多个指标；权重由 config.yaml 中 objectives.weights 控制。
    return (
        float(weights.get("sharpe", 0.0)) * g("sharpe")
        + float(weights.get("annual_return", 0.0)) * g("annual_return")
        + float(weights.get("max_drawdown", 0.0)) * g("max_drawdown")
        + float(weights.get("turnover", 0.0)) * g("turnover")
    )
