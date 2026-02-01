import os

os.environ["MPLBACKEND"] = "Agg"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CurvePlotPaths:
    equity_curve_png: str


def _set_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = [name]
            break

    matplotlib.rcParams["axes.unicode_minus"] = False


def _date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).date().isoformat()


def max_drawdown_window(equity: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return None, None

    equity.index = pd.to_datetime(equity.index)
    equity = equity.sort_index()

    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_ts = equity.loc[:trough].idxmax()
    return pd.Timestamp(peak_ts), pd.Timestamp(trough)


def plot_equity_curve_with_mdd(
    equity: pd.Series,
    out_path: str | Path,
    title: str = "回测收益曲线",
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_name: str = "沪深300ETF",
    periods_per_year: int = 252,
) -> CurvePlotPaths:
    _set_chinese_font()

    equity = pd.Series(equity).dropna()
    equity.index = pd.to_datetime(equity.index)
    equity = equity.sort_index()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    peak_ts, trough_ts = max_drawdown_window(equity)

    base = float(equity.iloc[0]) if len(equity) else 1.0
    cum_ret = equity / base - 1.0

    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else 0.0
    years = (len(equity) - 1) / float(periods_per_year) if len(equity) >= 2 else 0.0
    ann_ret = (float(equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    mdd = 0.0
    if peak_ts is not None and trough_ts is not None:
        try:
            mdd = float(equity.loc[trough_ts] / equity.loc[peak_ts] - 1.0)
        except Exception:
            mdd = 0.0

    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=150)

    ax.plot(cum_ret.index, cum_ret.values, color="#2F5597", linewidth=1.5, label="策略累计收益")
    ax.axhline(0.0, color="#999999", linewidth=0.8)

    # 基准：按 close/净值序列计算累计收益后绘制
    if benchmark_equity is not None and len(getattr(benchmark_equity, "index", [])) > 0:
        bser = pd.Series(benchmark_equity).dropna()
        bser.index = pd.to_datetime(bser.index)
        bser = bser.sort_index().reindex(equity.index)

        # 优先用 ffill 对齐，但如果开头仍为空，说明基准起点晚于策略起点；此时用 bfill 补齐开头
        bser = bser.ffill().bfill()

        if len(bser) > 0 and float(bser.iloc[0]) != 0.0 and pd.notna(bser.iloc[0]):
            bret = bser / float(bser.iloc[0]) - 1.0
            ax.plot(bret.index, bret.values, color="#C55A11", linewidth=1.2, label=f"基准：{benchmark_name}")
        else:
            ax.text(0.99, 0.01, "基准数据为空", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#C55A11")
    else:
        ax.text(0.99, 0.01, "未获取到基准数据", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#C55A11")

    if peak_ts is not None and trough_ts is not None and peak_ts <= trough_ts:
        ax.axvspan(peak_ts, trough_ts, color="#C00000", alpha=0.12, label="最大回撤区间")
        ax.scatter([peak_ts, trough_ts], [cum_ret.loc[peak_ts], cum_ret.loc[trough_ts]], color="#C00000", s=18)

        ax.annotate(
            f"开始: {_date_str(peak_ts)}",
            xy=(peak_ts, float(cum_ret.loc[peak_ts])),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#C00000",
        )
        ax.annotate(
            f"结束: {_date_str(trough_ts)}",
            xy=(trough_ts, float(cum_ret.loc[trough_ts])),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#C00000",
        )

    info_lines = [
        f"总收益率: {total_ret * 100:.2f}%",
        f"年化收益率: {ann_ret * 100:.2f}%",
    ]
    if peak_ts is not None and trough_ts is not None:
        info_lines.append(f"最大回撤: {abs(mdd) * 100:.2f}%")
    if peak_ts is not None and trough_ts is not None:
        info_lines.append(f"回撤区间: {_date_str(peak_ts)} ~ {_date_str(trough_ts)}")

    # 把统计信息放到图外（下方注释），避免与图例/曲线遮挡
    fig.text(
        0.01,
        -0.04,
        " | ".join(info_lines),
        va="bottom",
        ha="left",
        fontsize=10,
    )

    ax.set_title(title)
    ax.set_ylabel("累计收益")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return CurvePlotPaths(equity_curve_png=str(out_path))
