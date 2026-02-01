"""strategy/base_strategy.py

策略基类。

该项目中的策略负责：
- 接收行情数据（通常是包含 `close` 的 DataFrame）
- 生成与行情 index 对齐的目标仓位/权重（单标的可用 Series，多标的可用 DataFrame）

回测引擎会将仓位/权重与标的收益率相乘得到策略收益。
"""


class BaseStrategy:
    """策略基类，定义统一的参数注入与 positions 生成接口。"""

    def __init__(self, params: dict):
        # params 通常来自 strategy/strategy_params.py 中的 PARAMS dict。
        self.params = params

    def generate_positions(self, data):
        """根据行情数据生成仓位序列。

        参数:
            data: 行情数据（DataFrame/类似结构），由具体策略决定字段要求。

        返回:
            positions（一般为 pd.Series），需要能与 data 的时间索引对齐。

        子类必须实现该方法。
        """

        raise NotImplementedError
