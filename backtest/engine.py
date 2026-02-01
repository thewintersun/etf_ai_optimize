"""backtest/engine.py


支持两种模式：
- 单标的：data 包含 close 列，positions 为 Series
- 多标的：data 为 close 矩阵（columns=symbol），positions 为 DataFrame（columns=symbol）

注意：
- positions 表示“目标仓位/风险敞口/资金权重”，默认按 shift(1) 处理为 T+1 生效，以避免未来函数。
- 手续费使用简化模型：按换手量（仓位变化绝对值之和）乘以 fee_bps（基点）计算。
"""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.metrics import compute_metrics
from backtest.plotting import plot_equity_curve_with_mdd
from backtest.reporting import write_backtest_reports


def _apply_fees_single(returns: pd.Series, positions: pd.Series, fee_bps: float) -> pd.Series:
    fee = positions.diff().abs().fillna(0.0) * (fee_bps * 1e-4)
    return returns - fee


def _apply_fees_multi(returns: pd.Series, positions: pd.DataFrame, fee_bps: float) -> pd.Series:
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    fee_rate = turnover * (fee_bps * 1e-4)
    return returns - fee_rate


def _apply_slippage_multi(
    returns: pd.Series,
    close: pd.DataFrame,
    positions: pd.DataFrame,
    slippage: float,
    initial_capital: float,
) -> pd.Series:
    """估算并扣除滑点成本（按固定金额/每股）。"""
    if slippage <= 0.0:
        return returns

    # 换手权重
    turnover_w = positions.diff().abs().sum(axis=1).fillna(0.0)

    # 每日权益（用前一日权益作为当日交易规模基准）
    equity = (1.0 + returns).cumprod() * initial_capital
    equity_prev_day = equity.shift(1).fillna(initial_capital)

    # 当日换手总市值
    turnover_value = turnover_w * equity_prev_day

    # 用当日平均股价估算成交股数（这是一个简化，因为我们不希望在引擎里做复杂的逐笔计算）
    avg_price = close.mean(axis=1)
    # 避免除以 0
    avg_price[avg_price < 1e-6] = 1e-6

    # 估算的成交股数
    turnover_shares = turnover_value / avg_price

    # 总滑点成本（元）
    slippage_cost = turnover_shares * slippage

    # 滑点成本率
    slippage_rate = (slippage_cost / equity_prev_day).fillna(0.0)

    return returns - slippage_rate


def run_backtest(
    data: pd.DataFrame,
    positions,
    fee_bps: float = 0.0,
    slippage: float = 0.0,
    initial_capital: float = 1_000_000.0,
    periods_per_year: int = 252,
    write_reports: bool = True,
    report_dir: str | Path = "reports",
    lot_size: int = 100,
    symbol_name_map: dict | None = None,
    benchmark_close: pd.Series | None = None,
    benchmark_name: str = "沪深300ETF",
) -> dict:
    if isinstance(positions, pd.DataFrame):
        # 多标的模式：data 期望就是 close 矩阵
        # close：多标的收盘价矩阵（index=交易日期，columns=标的）
        # - astype(float)：确保后续 pct_change/乘法运算不会因为 dtype=object 而出错
        close = data.astype(float)

        # ret：每日简单收益率矩阵
        # - pct_change()：计算 close_t / close_{t-1} - 1
        # - fillna(0.0)：首日没有前值，收益率记为 0（不让 NaN 影响累计净值）
        ret = close.pct_change().fillna(0.0)

        # 将仓位表对齐到价格的交易日
        # - reindex(close.index)：如果 positions 缺失某些交易日，用价格的交易日作为“权威时间轴”
        # - fillna(0.0)：缺失日期默认空仓（0 权重/0 仓位）
        positions = positions.reindex(close.index).fillna(0.0)

        # T+1 生效（避免未来函数）：
        # - positions 通常表示“当日收盘后生成/调整的目标仓位”
        # - 交易执行在下一交易日，因此用 shift(1) 把仓位向后推一天作为当日实际持仓权重 w
        # - fillna(0.0)：回测第一天没有前一日仓位，默认空仓
        w = positions.shift(1).fillna(0.0)

        # strat_ret：策略当日总收益率（组合层面）
        # - (w * ret)：逐标的“权重 × 标的收益率”得到每个标的对组合的贡献
        # - sum(axis=1)：对所有标的求和得到组合当日收益率
        strat_ret = (w * ret).sum(axis=1)

        # 扣除手续费（多标的版）：
        # - 手续费基于 positions 的“换手/调仓幅度”（diff().abs() 的行和）
        # - fee_bps 为基点（bps），1bps=0.01%，函数内部会做 bps->比例 的换算
        # - 返回值是“净收益率序列”（已扣费）
        strat_ret = _apply_fees_multi(strat_ret, positions, float(fee_bps))

        # 扣除滑点（多标的版，按固定金额/每股估算）：
        # - slippage 单位：元/股（或元/份额）
        # - 由于引擎层只有“权重”没有逐笔成交，这里用换手对应的成交市值 / 平均价格 来粗略估算成交股数
        strat_ret = _apply_slippage_multi(
            strat_ret,
            close=close,
            positions=positions,
            slippage=float(slippage),
            initial_capital=float(initial_capital),
        )

        # equity：权益曲线/净值曲线（以 initial_capital 为起始资金）
        # - (1 + strat_ret).cumprod()：把每日收益率连乘得到累计收益倍数
        # - 乘以 initial_capital 得到以货币计的资金曲线
        equity = float(initial_capital) * (1.0 + strat_ret).cumprod()

        # 计算绩效指标（如年化收益/波动/夏普/最大回撤等，具体取决于 compute_metrics 实现）
        # - equity：资金曲线
        # - returns：日度净收益率
        # - positions：原始目标仓位（可用于统计换手、持仓暴露等）
        out = compute_metrics(equity=equity, returns=strat_ret, positions=positions, periods_per_year=periods_per_year)

        bench_equity = None
        if benchmark_close is not None and len(getattr(benchmark_close, "index", [])) > 0:
            bclose = pd.Series(benchmark_close).dropna()
            bclose.index = pd.to_datetime(bclose.index)
            bclose = bclose.sort_index().reindex(close.index).ffill()
            if len(bclose) > 0:
                bench_equity = bclose

        # 收益曲线图片（标注最大回撤区间 + 基准）
        curve_path = Path(report_dir) / (
            f"equity_curve_{pd.Timestamp(close.index.min()).date().isoformat()}_{pd.Timestamp(close.index.max()).date().isoformat()}.png"
        )
        cp = plot_equity_curve_with_mdd(
            equity=equity,
            benchmark_equity=bench_equity,
            benchmark_name=str(benchmark_name),
            periods_per_year=int(periods_per_year),
            out_path=curve_path,
            title="回测收益曲线",
        )
        out["equity_curve_png"] = cp.equity_curve_png

        # 报告：逐笔调仓成交 & 逐日持仓收益明细（每次运行生成独立文件）
        if write_reports:
            rp = write_backtest_reports(
                close=close,
                positions=positions,
                fee_bps=float(fee_bps),
                report_dir=report_dir,
                initial_capital=float(initial_capital),
                lot_size=int(lot_size),
                symbol_name_map=symbol_name_map,
            )
            out["trades_report_csv"] = rp.trades_csv
            out["daily_positions_report_csv"] = rp.daily_csv

        return out

    # 单标的模式（兼容原接口）
    close = data["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    positions = positions.reindex(close.index).fillna(0.0)
    strat_ret = positions.shift(1).fillna(0.0) * ret
    strat_ret = _apply_fees_single(strat_ret, positions, float(fee_bps))

    equity = float(initial_capital) * (1.0 + strat_ret).cumprod()
    out = compute_metrics(equity=equity, returns=strat_ret, positions=positions, periods_per_year=periods_per_year)

    bench_equity = None
    if benchmark_close is not None and len(getattr(benchmark_close, "index", [])) > 0:
        bclose = pd.Series(benchmark_close).dropna()
        bclose.index = pd.to_datetime(bclose.index)
        bclose = bclose.sort_index().reindex(close.index).ffill()
        if len(bclose) > 0:
            bench_equity = float(initial_capital) * (bclose / float(bclose.iloc[0]))

    curve_path = Path(report_dir) / (
        f"equity_curve_{pd.Timestamp(close.index.min()).date().isoformat()}_{pd.Timestamp(close.index.max()).date().isoformat()}.png"
    )
    cp = plot_equity_curve_with_mdd(
        equity=equity,
        benchmark_equity=bench_equity,
        benchmark_name=str(benchmark_name),
        periods_per_year=int(periods_per_year),
        out_path=curve_path,
        title="回测收益曲线",
    )
    out["equity_curve_png"] = cp.equity_curve_png

    if write_reports:
        # 单标的报告：把 close Series 转成矩阵以复用同一套报告逻辑
        close_df = close.to_frame(name="asset")
        pos_df = positions.to_frame(name="asset")
        rp = write_backtest_reports(
            close=close_df,
            positions=pos_df,
            fee_bps=float(fee_bps),
            report_dir=report_dir,
            initial_capital=float(initial_capital),
            lot_size=int(lot_size),
        )
        out["trades_report_csv"] = rp.trades_csv
        out["daily_positions_report_csv"] = rp.daily_csv

    return out
