"""backtest/run_backtest.py

回测脚本入口（命令行/批处理使用）。

该脚本的职责是：
- 生成一个交易日历（index）。
- 用 `strategy.strategy_params.PARAMS` 初始化策略。
- 策略内部通过 `data/manager.py` 拉取多 ETF 日线数据并生成权重（positions）。
- 调用 `backtest.engine.run_backtest` 计算回测指标（metrics）。
- 将 metrics 以 JSON 形式写入到 `metrics.json`，供 agent 在每轮迭代中读取并评分。

说明：
- 对于多标的策略，本脚本不再强制依赖 `data/processed/prices.csv`。
- 你可以通过 PARAMS 配置 start_date/end_date/provider/data_dir 等。
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import run_backtest
from strategy.joinquant_epo_strategy import JoinquantEPOStrategy
from strategy.strategy_params import PARAMS


def _make_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    # 简化：使用工作日作为交易日历；具体交易日缺口由 DataManager 拉取数据后自然对齐。
    return pd.date_range(start_date, end_date, freq="B")


def _setup_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """配置 root logger，使所有模块的 logger（包括 strategy.*）都能写入同一份日志。

    之前只给名为 `backtest` 的 logger 加 handler，会导致其它 logger（例如
    `strategy.joinquant_epo`）默认传播到 root，但 root 没有 handler，于是看不到输出。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加 handler（例如在交互式环境/重复运行时）
    if root.handlers:
        return logging.getLogger("backtest")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    # 仍然返回 backtest logger 作为入口 logger 使用
    return logging.getLogger("backtest")


def main() -> None:
    start_date = str(PARAMS.get("start_date", "2020-01-01"))
    end_date = str(PARAMS.get("end_date", pd.Timestamp.today().date().isoformat()))

    log_path = Path(PARAMS.get("log_file", "logs/backtest.log"))
    log_level_str = str(PARAMS.get("log_level", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = _setup_logging(log_path=log_path, level=log_level)

    logger.info("回测开始 | start_date=%s end_date=%s", start_date, end_date)
    logger.info("参数概览 | provider=%s data_dir=%s etf_pool_size=%s", PARAMS.get("provider"), PARAMS.get("data_dir"), len(PARAMS.get("etf_pool", [])))

    t0 = time.perf_counter()
    idx = _make_calendar(start_date, end_date)
    calendar_df = pd.DataFrame(index=idx)
    logger.info("交易日历生成完成 | n_days=%s from=%s to=%s", len(idx), idx.min().date().isoformat(), idx.max().date().isoformat())
    t1 = time.perf_counter()
    strat = JoinquantEPOStrategy(PARAMS)
    logger.info("策略初始化完成 | elapsed=%.3fs", (time.perf_counter() - t1))

    t2 = time.perf_counter()
    logger.info("生成持仓权重开始")
    weights = strat.generate_positions(calendar_df)
    logger.info("生成持仓权重完成 | shape=%s elapsed=%.3fs", getattr(weights, "shape", None), (time.perf_counter() - t2))

    # 回测引擎在多标的模式下，data 需要是 close 矩阵。
    # 策略内部使用同一套 DataManager 拉取 close，因此这里复用它生成 close 矩阵。
    t3 = time.perf_counter()
    logger.info("拉取收盘价矩阵开始")
    close = strat._fetch_close_matrix(start_date, end_date)  # noqa: SLF001
    logger.info("拉取收盘价矩阵完成 | shape=%s elapsed=%.3fs", getattr(close, "shape", None), (time.perf_counter() - t3))

    close = close.reindex(idx).sort_index().ffill()
    logger.info("收盘价对齐到交易日历 | shape=%s", close.shape)

    t4 = time.perf_counter()
    logger.info("回测引擎开始计算")
    benchmark_symbol = str(PARAMS.get("benchmark_symbol", "510300.SH"))

    benchmark_close = pd.Series(dtype=float)
    try:
        bdf = strat.dm.get_range(benchmark_symbol, start_date, end_date)
        if bdf is not None and (not bdf.empty) and ("close" in bdf.columns):
            benchmark_close = bdf["close"].astype(float)
    except Exception:
        logger.exception("拉取基准收盘价失败 | sym=%s", benchmark_symbol)
    
    metrics = run_backtest(
        data=close,
        positions=weights,
        fee_bps=float(PARAMS.get("fee_bps", 0.0)),
        slippage=float(PARAMS.get("slippage", 0.0)),
        initial_capital=float(PARAMS.get("initial_capital", 1_000_000.0)),
        periods_per_year=252,
        lot_size=int(PARAMS.get("lot_size", 100)),
        symbol_name_map=PARAMS.get("symbol_name_map", {}),
        benchmark_close=benchmark_close,
        benchmark_name=str(PARAMS.get("symbol_name_map", {}).get(benchmark_symbol, benchmark_symbol)),
    )
    logger.info("回测引擎计算完成 | elapsed=%.3fs", (time.perf_counter() - t4))

    out_path = Path("metrics.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("回测指标已写入 | path=%s elapsed_total=%.3fs", str(out_path), (time.perf_counter() - t0))

    if isinstance(metrics, dict) and metrics.get("equity_curve_png"):
        logger.info("收益曲线图片已生成 | path=%s", metrics.get("equity_curve_png"))


if __name__ == "__main__":
    main()
