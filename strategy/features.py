"""strategy/features.py

特征/指标函数集合。

该模块提供给策略与回测使用的一些基础时间序列计算：
- momentum: 动量（n 期收益）。
- volatility: 波动率（收益率 rolling std）。
- max_drawdown: 最大回撤（返回负值，例如 -0.2 表示回撤 20%）。

说明：
- 这些函数尽量保持无状态、纯函数，便于复用与测试。
- `_to_series` 做了轻量的输入适配，允许传入 list/ndarray 等。
"""

import pandas as pd


def _to_series(x):
    """将输入转换为 pandas.Series。

    之所以做这个适配，是因为有些调用方可能传入 numpy array 或 list。
    """

    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def momentum(close: pd.Series, window: int) -> pd.Series:
    """计算动量：window 期收益率。"""

    close = _to_series(close)
    return close.pct_change(window)


def volatility(close: pd.Series, window: int) -> pd.Series:
    """计算波动率：收益率 rolling 标准差。"""

    close = _to_series(close)
    ret = close.pct_change()
    return ret.rolling(window).std(ddof=0)


def max_drawdown(equity: pd.Series) -> float:
    """计算最大回撤。

    返回:
        一个非正数（<= 0）。例如 -0.2 表示从峰值到谷底最大回撤为 20%。
    """

    equity = _to_series(equity).dropna()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())
