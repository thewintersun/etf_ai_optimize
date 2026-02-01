"""backtest/metrics.py

回测指标计算。

该模块把资金曲线（equity）、策略收益（returns）、仓位序列（positions）转换为
一组可序列化的指标 dict。

指标说明（当前实现）：
- annual_return: 年化收益率（CAGR 近似）。
- max_drawdown: 最大回撤（正值，越小越好）。
- sharpe: 夏普比率（未扣无风险利率）。
- calmar: Calmar = annual_return / max_drawdown。
- turnover: 简化换手（单标的为仓位变化绝对值求和；多标的为各资产换手绝对值之和再按时间求和）。
"""

from __future__ import annotations

import math

import pandas as pd

from strategy.features import max_drawdown


def annual_return(equity: pd.Series, periods_per_year: int) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    years = (len(equity) - 1) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return total ** (1.0 / years) - 1.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    mu = float(returns.mean())
    sd = float(returns.std(ddof=0))
    if sd == 0.0:
        return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)


def turnover(positions) -> float:
    positions = positions.fillna(0.0)
    if isinstance(positions, pd.DataFrame):
        return float(positions.diff().abs().sum(axis=1).sum())
    return float(positions.diff().abs().sum())


def compute_metrics(equity: pd.Series, returns: pd.Series, positions, periods_per_year: int) -> dict:
    mdd = -max_drawdown(equity)
    ann = annual_return(equity, periods_per_year)
    shrp = sharpe_ratio(returns, periods_per_year)
    t = turnover(positions)
    calmar = (ann / mdd) if mdd > 0 else 0.0
    return {
        "annual_return": float(ann),
        "max_drawdown": float(mdd),
        "sharpe": float(shrp),
        "calmar": float(calmar),
        "turnover": float(t),
    }
