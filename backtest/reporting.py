from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReportPaths:
    trades_csv: str
    daily_csv: str


def _safe_date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _run_id_now() -> str:
    # 例：20260131_150322
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_backtest_reports(
    close: pd.DataFrame,
    positions: pd.DataFrame,
    fee_bps: float = 0.0,
    report_dir: str | Path = "reports",
    run_id: Optional[str] = None,
    initial_capital: float = 1_000_000.0,
    lot_size: int = 100,
    symbol_name_map: Optional[Dict[str, str]] = None,
) -> ReportPaths:
    """把回测期内的“逐笔调仓成交”和“逐日持仓收益明细”写入两个 CSV 文件。

    重要约定（与 backtest.engine.run_backtest 保持一致）：
    - `positions` 为“目标权重/风险敞口”，回测收益使用 T+1 生效（w=positions.shift(1)）
    - 本报告把“调仓成交”视为在当日收盘价发生：把当前持仓调整到 positions.loc[date]
    - 手续费仍用引擎的简化模型：fee_rate = turnover * fee_bps * 1e-4
      其中 turnover = sum(|positions_t - positions_{t-1}|)，并按“当日开盘前组合权益”计费
    """

    if close is None or close.empty:
        raise ValueError("close is empty")

    close = close.copy()
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    if positions is None or len(getattr(positions, "index", [])) == 0:
        raise ValueError("positions is empty")

    # 对齐 index / columns
    positions = positions.copy()
    positions.index = pd.to_datetime(positions.index)
    positions = positions.reindex(close.index).fillna(0.0)
    positions = positions.reindex(columns=close.columns, fill_value=0.0).astype(float)

    # 基础信息
    start = _safe_date_str(close.index.min())
    end = _safe_date_str(close.index.max())
    run_id = run_id or _run_id_now()

    report_dir_path = Path(report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)

    trades_path = report_dir_path / f"trades_{start}_{end}_{run_id}.csv"
    daily_path = report_dir_path / f"daily_positions_{start}_{end}_{run_id}.csv"

    syms = list(close.columns)
    shares = pd.Series(0.0, index=syms, dtype=float)
    avg_cost = pd.Series(0.0, index=syms, dtype=float)
    cash = float(initial_capital)  # 初始资金（元）

    trade_rows: List[Dict[str, float | str]] = []
    daily_rows: List[Dict[str, float | str]] = []

    prev_prices: Optional[pd.Series] = None
    prev_target_w = pd.Series(0.0, index=syms, dtype=float)

    eps = 1e-12
    fee_bps = float(fee_bps)

    for i, dt in enumerate(close.index):
        px = close.loc[dt].astype(float)
        if prev_prices is None:
            prev_prices = px.copy()

        # 组合“日初权益”：用昨日收盘价估值（与收益计算口径一致）
        equity_start = float(cash + float((shares * prev_prices).sum()))

        # 先走到当日收盘：持仓因价格变化产生当日盈亏
        daily_pnl_per_sym = shares * (px - prev_prices)
        equity_before_fee = float(cash + float((shares * px).sum()))

        # 今日调仓带来的手续费（按引擎简化模型）
        target_w = positions.loc[dt].astype(float)
        delta_target_w = (target_w - prev_target_w).astype(float)
        turnover = float(delta_target_w.abs().sum())
        fee_rate = turnover * (fee_bps * 1e-4)
        fee_amount_total = float(equity_start * fee_rate)

        # 扣除手续费（假设从现金中扣）
        if fee_amount_total != 0.0:
            cash -= fee_amount_total

        equity_after_fee = float(equity_before_fee - fee_amount_total)

        # 输出当日（持仓=当日持仓，即 w=positions.shift(1) 口径的真实持仓）
        # 注：shares/avg_cost 此时尚未执行“今日收盘调仓”，因此与引擎收益口径一致。
        total_unrealized = float(((shares * px) - (shares * avg_cost)).sum())
        total_daily_pnl = float(daily_pnl_per_sym.sum())

        for sym in syms:
            qty = float(shares[sym])
            if abs(qty) <= eps:
                # 空仓也记一行会很长；按常见需求只记录有持仓的品种
                continue

            close_px = float(px[sym])
            mv = float(qty * close_px)
            cost = float(qty * float(avg_cost[sym]))
            unreal = float(mv - cost)
            daily_pnl = float(daily_pnl_per_sym[sym])

            pos_ratio = float(mv / equity_after_fee) if equity_after_fee != 0.0 else 0.0
            pnl_ratio = float(unreal / total_unrealized) if total_unrealized != 0.0 else 0.0

            daily_rows.append(
                {
                    "日期": _safe_date_str(dt),
                    "品种": str(sym),
                    "品种名称": str(symbol_name_map.get(str(sym), "")) if isinstance(symbol_name_map, dict) else "",
                    "持有数量": qty,
                    "当天收盘价": close_px,
                    "市值": mv,
                    "品种盈亏": unreal,
                    "开仓均价": float(avg_cost[sym]),
                    "当日盈亏": daily_pnl,
                    "仓位占比": pos_ratio,
                    "盈亏占比": pnl_ratio,
                }
            )

        # 今日收盘执行调仓：
        # - 只在目标权重发生变化时调仓（更贴近“每次调仓”的语义，也避免非调仓日产生大量微调成交）
        # - 用“扣费后的权益”作为目标规模，成交价按收盘价。
        if turnover > eps:
            target_values = target_w * equity_after_fee
            target_shares = target_values / px
            target_shares = target_shares.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

            # A股/ETF 常见最小交易单位：100股（1手）。这里强制目标持仓数量为 lot_size 的整数倍。
            lot = int(lot_size) if int(lot_size) > 0 else 1
            target_shares = (np.floor(target_shares / lot) * lot).astype(float)

            # 先计算每个品种的成交额，用于手续费按成交额分摊
            trade_amount_by_sym: Dict[str, float] = {}
            for sym in syms:
                cur = float(shares[sym])
                tgt = float(target_shares[sym])
                dsh = tgt - cur
                if abs(dsh) <= eps:
                    continue
                trade_amount_by_sym[sym] = float(abs(dsh) * float(px[sym]))

            total_trade_amount = float(sum(trade_amount_by_sym.values()))

            # 逐品种生成成交（仅记录变化的标的）
            for sym in syms:
                cur = float(shares[sym])
                tgt = float(target_shares[sym])
                dsh = tgt - cur
                if abs(dsh) <= eps:
                    continue

                price = float(px[sym])
                amount = float(abs(dsh) * price)
                side = "买" if dsh > 0 else "卖"

                fee_alloc = 0.0
                if total_trade_amount > 0:
                    fee_alloc = float(fee_amount_total * (float(trade_amount_by_sym.get(sym, 0.0)) / total_trade_amount))

                # 平仓盈亏：仅在卖出时计算（按当前均价作为成本）
                realized = 0.0
                if dsh < 0:
                    sell_qty = float(-dsh)
                    realized = float((price - float(avg_cost[sym])) * sell_qty)

                trade_rows.append(
                    {
                        "日期": _safe_date_str(dt),
                        "时间": "15:00:00",
                        "品种": str(sym),
                        "品种名称": str(symbol_name_map.get(str(sym), "")) if isinstance(symbol_name_map, dict) else "",
                        "交易类型": side,
                        "成交数量": float(abs(dsh)),
                        "成交价": price,
                        "成交额": amount,
                        "平仓盈亏": realized,
                        "手续费": fee_alloc,
                    }
                )

                # 更新现金
                cash -= float(dsh * price)  # 买入扣现金；卖出增加现金（dsh<0）

                # 更新均价/持仓
                if dsh > 0:
                    # 加仓：按加权平均更新均价
                    new_qty = cur + float(dsh)
                    if new_qty > eps:
                        avg_cost[sym] = float((cur * float(avg_cost[sym]) + float(dsh) * price) / new_qty)
                    else:
                        avg_cost[sym] = 0.0
                    shares[sym] = new_qty
                else:
                    # 减仓：均价不变；清仓则均价归零
                    new_qty = cur + float(dsh)
                    shares[sym] = new_qty
                    if abs(new_qty) <= eps:
                        shares[sym] = 0.0
                        avg_cost[sym] = 0.0

        # 推进到下一天
        prev_prices = px
        prev_target_w = target_w

    # 写文件（utf-8-sig 便于 Excel 直接打开）
    trades_df = pd.DataFrame(trade_rows)
    daily_df = pd.DataFrame(daily_rows)

    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=["日期", "时间", "品种", "品种名称", "交易类型", "成交数量", "成交价", "成交额", "平仓盈亏", "手续费"]
        )
    if daily_df.empty:
        daily_df = pd.DataFrame(
            columns=["日期", "品种", "品种名称", "持有数量", "当天收盘价", "市值", "品种盈亏", "开仓均价", "当日盈亏", "仓位占比", "盈亏占比"]
        )

    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    return ReportPaths(trades_csv=str(trades_path), daily_csv=str(daily_path))

