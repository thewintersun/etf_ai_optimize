import datetime
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.linalg import solve
from scipy.stats import linregress
from sklearn.covariance import LedoitWolf

from data.manager import get_manager
from strategy.base_strategy import BaseStrategy


logger = logging.getLogger("strategy.joinquant_epo")


def dynamic_w_ledoitwolf(returns: pd.DataFrame) -> float:
    """根据样本收益率自适应估计 Ledoit-Wolf 协方差收缩强度 w。

    这里的 `w` 用于在相关矩阵/协方差矩阵上做 shrinkage：
    - w 越大：越偏向对角阵（资产间相关性被更强地压缩），组合更“稳健”但更保守
    - w 越小：越接近样本相关结构，可能更敏感/更易过拟合

    参数:
        returns: shape=(T, N) 的收益率矩阵（行是时间，列是资产）

    返回:
        float: 估计得到的 shrinkage 强度，通常在 [0, 1]
    """
    lw = LedoitWolf().fit(returns.values)
    return float(lw.shrinkage_)


def garch_vol_forecast(series: pd.Series) -> float:
    """用 GARCH(1,1) 对单资产收益率做一步波动率预测。

    注意:
        - `arch` 对输入数据的尺度敏感：若收益率量级过小会频繁触发 DataScaleWarning，
          并可能影响收敛；因此这里做一个简单的缩放再换算回原尺度。
        - 输入为收益率序列（通常是日收益率），输出为预测的标准差（波动率）。
        - 若序列为空/拟合失败，返回 NaN（上层会回退到等权 anchor）。

    参数:
        series: 单资产收益率序列

    返回:
        float: 一步预测波动率（标准差）
    """
    series = series.dropna()
    if series.empty:
        return float("nan")

    # DataScaleWarning: y is poorly scaled...
    # 对收益率做缩放以改善数值稳定性（例如日收益率 ~1e-5 的情况）。
    scale = 100.0
    y = series.astype(float) * scale

    try:
        am = arch_model(y, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off")
        fcast = res.forecast(horizon=1).variance.values[-1][0]
        # variance 是缩放后的方差，需要除以 scale^2；std 再除以 scale。
        return float(np.sqrt(fcast)) / scale
    except Exception:
        logger.exception("GARCH 波动率预测失败")
        return float("nan")


def dynamic_anchor_garch(returns: pd.DataFrame) -> np.ndarray:
    """使用各资产 GARCH 预测波动率构造“风险平价”锚定权重 anchor。

    思路:
        - 对每个资产分别预测下一期波动率 vol
        - 以 1/vol 作为风险平价近似（波动小的资产占比更高）
        - 归一化后得到 anchor，作为 EPO 的锚点权重

    失败回退:
        - 若所有资产都无法得到有效波动率（全 NaN 或 <=0），回退到等权。

    参数:
        returns: shape=(T, N) 的收益率矩阵

    返回:
        np.ndarray: shape=(N,) 的 anchor 权重（和为 1）
    """
    vols: List[float] = []
    for c in returns.columns:
        vols.append(garch_vol_forecast(returns[c]))
    vols_arr = np.array(vols, dtype=float)

    vols_arr = np.where(np.isfinite(vols_arr) & (vols_arr > 0), vols_arr, np.nan)
    if np.all(np.isnan(vols_arr)):
        return np.ones(len(returns.columns), dtype=float) / len(returns.columns)

    # Risk parity anchor
    inv_vol = 1.0 / vols_arr
    inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)
    if inv_vol.sum() <= 0:
        return np.ones(len(returns.columns), dtype=float) / len(returns.columns)

    anchor = inv_vol / inv_vol.sum()
    return anchor


def ts_sharpe_signal(prices: pd.DataFrame, span: int = 60) -> np.ndarray:
    """时间序列（TS）Sharpe 风格的信号：EWMA(mean)/EWMA(std)。

    用途:
        - 当基础信号退化（比如均值/标准差得到全 0 或 NaN）时作为备选信号

    参数:
        prices: shape=(T, N) 的价格矩阵
        span: EWMA 衰减窗口（span 越大越平滑）

    返回:
        np.ndarray: shape=(N,) 的信号向量；无效值会被置为 0
    """
    returns = prices.pct_change().dropna()
    if returns.empty:
        return np.zeros(prices.shape[1], dtype=float)
    mean = returns.ewm(span=span).mean().iloc[-1]
    std = returns.ewm(span=span).std().iloc[-1]
    out = (mean / std).values
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def dynamic_lambda(port_vol_annual: float) -> float:
    """根据组合年化波动粗略自适应 EPO 风险厌恶系数 lambda。

    直觉:
        - 组合越“波动”，越倾向于加大 lambda（降低风险暴露，偏保守）

    参数:
        port_vol_annual: 组合年化波动率（例如由等权组合估算）

    返回:
        float: lambda
    """
    if port_vol_annual > 0.18:
        return 3.0
    elif port_vol_annual > 0.14:
        return 2.5
    else:
        return 2.0


def epo(
    x: pd.DataFrame,
    signal: np.ndarray,
    lambda_: float,
    method: str = "simple",
    w: Optional[float] = None,
    anchor: Optional[np.ndarray] = None,
    normalize: bool = True,
    endogenous: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """EPO（Expected Portfolio Optimization）核心实现。

    输入:
        - x: 收益率矩阵（T,N）
        - signal: 预期收益/打分信号（N,）
        - lambda_: 风险厌恶系数（越大越保守）
        - method:
            - "simple": 经典均值-方差形式：w ~ inv(cov) @ signal / lambda
            - "anchored": 引入锚点组合（anchor）与 shrinkage 的“稳健”形式
        - w: shrinkage 强度（通常来自 LedoitWolf），用于把相关结构往对角收缩
        - anchor: 锚点权重（N,），常用风险平价/等权等
        - normalize:
            - True: 强制非负并归一化到和为 1（会丢失杠杆信息）
            - False: 返回原始解（后续可自行 clip/归一化/目标波动缩放）
        - endogenous:
            - anchored 模式下用于计算 gamma 的开关

    关键矩阵:
        - vcov: 样本协方差
        - corr: 样本相关矩阵
        - V: 对角协方差（只保留各资产方差，相关置 0）
        - cov_tilde: 用 shrunk 相关矩阵重构的协方差

    返回:
        (weights, cov_tilde)
        - weights: shape=(N,) 的权重（可能未归一化，取决于 normalize）
        - cov_tilde: shrink 后用于后续风险估计/风险目标缩放的协方差

    备注:
        该实现与论文/实现细节可能略有差异，此处更偏“工程可用”的稳健版本。
    """
    n = x.shape[1]
    vcov = x.cov()
    corr = x.corr()
    I = np.eye(n)
    V = np.diag(np.diag(vcov.values))
    std = np.sqrt(V)
    s = signal.reshape(-1)
    a = anchor.reshape(-1) if anchor is not None else np.ones(n) / n

    if w is None:
        w = 0.0

    # 相关矩阵 shrink：
    #   w=0 -> 使用样本相关；w=1 -> 完全对角（单位阵）
    shrunk_cor = ((1 - w) * (I @ corr.values)) + (w * I)

    # 用 shrink 后的相关矩阵重构协方差（标准差矩阵 * corr * 标准差矩阵）
    cov_tilde = std @ shrunk_cor @ std

    # 额外保留一份“对角协方差”混合（有些实现会用到），这里主要用于 anchored 形式的项
    shrunk_cov = (1 - w) * cov_tilde + w * V

    # 计算 inv(cov_tilde)。这里用 solve(A, I) 比显式求逆更稳定。
    inv_shrunk_cov = solve(cov_tilde, I)

    if method == "simple":
        out = (1.0 / lambda_) * inv_shrunk_cov @ s
    elif method == "anchored":
        # anchored 版本：在信号与锚点之间做一种内生/外生的混合
        if endogenous:
            # gamma 用于让信号项与锚点项在“风险尺度”上可比，避免尺度不一致。
            num = float(np.sqrt(a.T @ cov_tilde @ a))
            den = float(np.sqrt(s.T @ inv_shrunk_cov @ cov_tilde @ inv_shrunk_cov @ s))
            gamma = (num / den) if (den > 0 and np.isfinite(den)) else 0.0

            # 注意：这里 I@V@a 等价于 V@a（对角协方差作用于锚点）
            out = inv_shrunk_cov @ (((1 - w) * gamma * s) + (w * (I @ V @ a)))
        else:
            out = inv_shrunk_cov @ (((1 - w) * (1.0 / lambda_) * s) + (w * (I @ V @ a)))
    else:
        raise ValueError("`method` not accepted. Try `simple` or `anchored` instead.")

    if normalize:
        # 工程约束：不做空（负权置 0），再归一化到和为 1
        out = np.array([0.0 if v < 0 else float(v) for v in out], dtype=float)
        ssum = out.sum()
        if ssum > 0:
            out = out / ssum

    return out, cov_tilde


def _normalize_symbol_for_manager(sym: str) -> str:
    """把 ETF 代码规范化为 DataManager 支持的格式。

    约束:
        - 必须包含交易所后缀，形如 "600000.SH" / "000001.SZ"（大小写不敏感）
        - 仅支持上交所/深交所：SH / SZ

    参数:
        sym: 原始 symbol 字符串（可能包含多余空格，后缀大小写不一致）

    返回:
        str: 规范化后的 symbol（如 "510300.SH"）
    """
    s = sym.strip()
    if not s:
        raise ValueError("empty symbol")
    if "." not in s:
        raise ValueError(f"invalid symbol: {sym}")
    code, ex = s.split(".", 1)
    exl = ex.lower()
    if exl not in ("sh", "sz"):
        raise ValueError(f"invalid exchange: {sym}")
    # 统一为 DataManager 支持的 A 股格式：600000.SH / 000001.SZ
    return f"{code}.{exl.upper()}"


@dataclass
class JoinquantEPOConfig:
    """Joinquant EPO 策略参数集合。

    字段说明:
        - etf_pool: ETF 代码池（会在 Strategy 初始化时做规范化）
        - rebalance_weekday: 调仓星期（注意：此策略用 dt.weekday()+1 参与比较）
        - data_window: 优化使用的回看窗口长度（交易日数量）
        - min_history_days: 单资产最少有效历史点数，不足则剔除
        - signal_span: 备选 TS Sharpe 信号的 EWMA span
        - w_clip_min/w_clip_max: 对 LedoitWolf shrinkage 强度做裁剪，避免极端值
        - use_dynamic_lambda: 是否使用动态 lambda（由组合波动估算），否则用 epo_lambda
        - epo_lambda: 固定 lambda（当 use_dynamic_lambda=False 时使用）
        - max_weight: 单一资产权重上限（None 表示不限制；仅在 long-only + normalize 后应用）
        - use_risk_target: 是否启用目标波动缩放（会引入杠杆/去杠杆）
        - target_vol_annual: 目标年化波动率
    """

    etf_pool: List[str]
    rebalance_weekday: int = 3
    data_window: int = 90
    min_history_days: int = 30
    # 优化/估计时实际使用的窗口长度上限（交易日数）。
    # 例如 data_window=90 但只想用最近 30 个交易日做协方差/信号估计，可设为 30。
    # 注意：仍会先用 min_history_days 做“历史足够”过滤。
    opt_tail_days: int = 30
    signal_span: int = 60
    w_clip_min: float = 0.2
    w_clip_max: float = 0.6
    use_dynamic_lambda: bool = True
    epo_lambda: float = 2.0
    max_weight: Optional[float] = None
    use_risk_target: bool = True
    target_vol_annual: float = 0.12
    max_leverage: float = 1.5
    signal_clip_quantile: float = 0.1
    signal_tanh_scale: float = 1.0
    signal_blend_alpha: float = 1.0
    rebalance_min_delta: float = 0.0
    rebalance_smoothing_beta: float = 1.0


class JoinquantEPOStrategy(BaseStrategy):
    """基于 EPO（Expected Portfolio Optimization）的 ETF 组合策略。

    数据流:
        - `generate_positions` 接收回测引擎给定的 `data`（仅用其 index 决定日期范围）
        - 从 DataManager 拉取 ETF 池的 close 序列并对齐到回测日历
        - 在指定调仓日，用最近 `data_window` 的价格（转收益率）做优化
        - 输出每个交易日的目标持仓权重矩阵（index=交易日，columns=ETF）

    重要约束/假设:
        - 默认不允许做空（最终对 raw 权重做 clip>=0）
        - 调仓日比较逻辑：`(dt.weekday() + 1) == rebalance_weekday`
          其中 `dt.weekday()` 为 Monday=0..Sunday=6，因此 `rebalance_weekday` 的取值约定为 1..7。
    """

    def __init__(self, params: dict):
        """从 `params` 构造策略并初始化数据管理器。"""
        super().__init__(params)
        logger.info("策略初始化 | provider=%s data_dir=%s", params.get("provider", "akshare"), params.get("data_dir", "data/cache"))
        self.dm = get_manager(provider=str(params.get("provider", "akshare")), data_dir=str(params.get("data_dir", "data/cache")))

        self.cfg = JoinquantEPOConfig(
            etf_pool=[_normalize_symbol_for_manager(s) for s in params.get("etf_pool", [])],
            rebalance_weekday=int(params.get("rebalance_weekday", 3)),
            data_window=int(params.get("data_window", 90)),
            min_history_days=int(params.get("min_history_days", 30)),
            opt_tail_days=int(params.get("opt_tail_days", 30)),
            signal_span=int(params.get("signal_span", 60)),
            w_clip_min=float(params.get("w_clip_min", 0.2)),
            w_clip_max=float(params.get("w_clip_max", 0.6)),
            use_dynamic_lambda=bool(params.get("use_dynamic_lambda", True)),
            epo_lambda=float(params.get("epo_lambda", 2.0)),
            max_weight=(None if params.get("max_weight", None) in (None, "", "None") else float(params.get("max_weight"))),
            use_risk_target=bool(params.get("use_risk_target", True)),
            target_vol_annual=float(params.get("target_vol_annual", 0.12)),
            max_leverage=float(params.get("max_leverage", 1.5)),
            signal_clip_quantile=float(params.get("signal_clip_quantile", 0.1)),
            signal_tanh_scale=float(params.get("signal_tanh_scale", 1.0)),
            signal_blend_alpha=float(params.get("signal_blend_alpha", 1.0)),
            rebalance_min_delta=float(params.get("rebalance_min_delta", 0.0)),
            rebalance_smoothing_beta=float(params.get("rebalance_smoothing_beta", 1.0)),
        )

    def _fetch_close_matrix(self, start: str, end: str) -> pd.DataFrame:
        """拉取 ETF 池在 [start, end] 的收盘价矩阵。

        参数:
            start/end: ISO 格式日期字符串（YYYY-MM-DD）

        返回:
            pd.DataFrame: index 为交易日，columns 为 ETF，值为 close。
            若无可用数据，返回空 DataFrame。
        """
        t0 = time.perf_counter()
        closes: Dict[str, pd.Series] = {}
        ok = 0
        skipped = 0

        logger.info("拉取收盘价矩阵开始 | start=%s end=%s pool_size=%s", start, end, len(self.cfg.etf_pool))

        for sym in self.cfg.etf_pool:
            try:
                df = self.dm.get_range(sym, start, end)
            except Exception:
                logger.exception("拉取收盘价失败 | sym=%s", sym)
                skipped += 1
                continue

            if df is None or df.empty or "close" not in df.columns:
                skipped += 1
                logger.warning("跳过拉取 | sym=%s reason=%s", sym, "empty_or_no_close")
                continue

            s = df["close"].astype(float)
            closes[sym] = s
            ok += 1

        if not closes:
            logger.warning("收盘价矩阵为空 | elapsed=%.3fs", (time.perf_counter() - t0))
            return pd.DataFrame()

        close_df = pd.concat(closes, axis=1).sort_index()
        close_df.columns = list(closes.keys())
        logger.info(
            "fetch_close_matrix done | ok=%s skipped=%s shape=%s elapsed=%.3fs",
            ok,
            skipped,
            close_df.shape,
            (time.perf_counter() - t0),
        )
        return close_df

    def _run_optimization(self, prices: pd.DataFrame) -> Dict[str, float]:
        """对给定价格窗口执行一次 EPO 优化，返回目标权重。

        主要步骤:
            1) 清洗价格矩阵（去掉全空行/任意含空的列）
            2) 计算收益率
            3) 用等权组合估一个 proxy 波动 -> 选择 lambda
            4) 构造信号：均值/标准差；若退化则用 TS Sharpe 备选
            5) LedoitWolf 得到 shrinkage 强度 w（并裁剪到区间）
            6) GARCH 预测波动 -> 构造风险平价 anchor
            7) 调用 `epo(..., method="anchored")` 得到 raw 权重
            8) clip 非负、归一化
            9) 可选：按目标波动率做整体缩放（引入杠杆系数）

        参数:
            prices: shape=(T, N) 的价格矩阵

        返回:
            Dict[str, float]: {symbol: weight}
            注意：如果开启风险目标缩放，权重之和可能不为 1（表示杠杆/去杠杆）。
        """
        t0 = time.perf_counter()
        prices = prices.dropna(how="all")
        prices = prices.dropna(axis=1, how="any")
        if prices.shape[1] == 0 or prices.shape[0] < 5:
            logger.debug("优化跳过：价格数据不足 | reason=%s shape=%s", "insufficient_prices", getattr(prices, "shape", None))
            return {}

        returns = prices.pct_change().dropna()
        if returns.empty:
            logger.debug("优化跳过：收益率为空 | reason=%s", "empty_returns")
            return {}

        # proxy portfolio vol (equal weight)
        ew = np.ones(len(returns.columns), dtype=float) / len(returns.columns)
        port_vol_daily = float(np.sqrt(ew.T @ returns.cov().values @ ew))
        port_vol_annual = port_vol_daily * np.sqrt(252)

        # 信号融合：横截面(mean/std) 与 时间序列(ts_sharpe)
        cs_sig = (returns.mean() / returns.std()).values
        cs_sig = np.where(np.isfinite(cs_sig), cs_sig, 0.0)

        ts_sig = ts_sharpe_signal(prices, span=int(self.cfg.signal_span))
        ts_sig = np.where(np.isfinite(ts_sig), ts_sig, 0.0)

        alpha = float(self.cfg.signal_blend_alpha)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        signal_std = (alpha * cs_sig) + ((1.0 - alpha) * ts_sig)

        # 若融合信号退化为 0，则回退到等权 0 信号（后续会被 clip/tanh 处理为 0）
        used_fallback_signal = False
        if float(np.abs(signal_std).sum()) == 0.0:
            used_fallback_signal = True

        # 信号去极值（winsorize/clip by quantile）+ tanh 压缩
        q = float(self.cfg.signal_clip_quantile)
        q = float(np.clip(q, 0.0, 0.49)) # Ensure quantile is valid
        if q > 0 and len(signal_std) > 1:
            lo = float(np.nanquantile(signal_std, q))
            hi = float(np.nanquantile(signal_std, 1.0 - q))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                signal_std = np.clip(signal_std, lo, hi)

        sscale = float(self.cfg.signal_tanh_scale)
        if np.isfinite(sscale) and sscale > 0:
            signal_std = np.tanh(signal_std / sscale)
        else:
            signal_std = np.tanh(signal_std)

        # 计算动态 lambda，但根据配置决定是否使用
        dynamic_lambda_val = dynamic_lambda(port_vol_annual)
        lambda_ = float(dynamic_lambda_val) if bool(self.cfg.use_dynamic_lambda) else float(self.cfg.epo_lambda)

        # 使用 LedoitWolf 的 shrinkage 强度，并进行裁剪，避免过度/不足收缩
        w_raw = dynamic_w_ledoitwolf(returns)
        w_dyn = float(np.clip(w_raw, self.cfg.w_clip_min, self.cfg.w_clip_max))

        anchor = dynamic_anchor_garch(returns)

        try:
            raw, shrunk_cov = epo(
                x=returns,
                signal=signal_std,
                lambda_=lambda_,  # 使用动态或配置的lambda值
                method="anchored",
                w=w_dyn,
                anchor=anchor,
                endogenous=True,
                normalize=False,
            )
        except Exception:
            logger.exception("EPO 优化失败")
            return {}

        # 工程约束：不做空
        raw = np.clip(raw, 0.0, None)
        if float(raw.sum()) <= 0:
            weights = np.ones(len(prices.columns), dtype=float) / len(prices.columns)
        else:
            weights = raw / float(raw.sum())

        # 单一资产权重上限（可选）：仅在 long-only 且归一化后应用
        if self.cfg.max_weight is not None:
            mw = float(self.cfg.max_weight)
            if mw > 0 and np.isfinite(mw):
                weights = np.clip(weights, 0.0, mw)
                ssum2 = float(weights.sum())
                if ssum2 > 0:
                    weights = weights / ssum2

        scaled = False
        scale = 1.0

        # 风险目标缩放：整体乘以 scale，使组合年化波动接近 target_vol_annual
        # 注意：这会使权重之和 != 1（相当于有杠杆/去杠杆）。
        if self.cfg.use_risk_target:
            port_vol_daily2 = float(np.sqrt(weights.T @ shrunk_cov @ weights))
            port_vol_annual2 = port_vol_daily2 * np.sqrt(252)
            if port_vol_annual2 > 0 and np.isfinite(port_vol_annual2):
                scale = float(self.cfg.target_vol_annual) / float(port_vol_annual2)
                # 杠杆上限：避免极端情况下 scale 过大
                if np.isfinite(scale) and scale > float(self.cfg.max_leverage):
                    scale = float(self.cfg.max_leverage)
                weights = weights * scale
                scaled = True

        '''
        logger.info(
            "run_optimization done | n_assets=%s used_fallback_signal=%s lambda=%.3f w=%.3f scaled=%s scale=%.3f elapsed=%.3fs",
            len(prices.columns),
            used_fallback_signal,
            float(lambda_),
            float(w_dyn),
            scaled,
            float(scale),
            (time.perf_counter() - t0),
        )
        '''

        return dict(zip(prices.columns.tolist(), weights.tolist()))

    def generate_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成回测期内每天的目标持仓权重。

        约定:
            - 输入 `data` 只要求有 datetime index（列内容不直接使用）
            - 输出 DataFrame 的 index 必须与输入 `data.index` 对齐
            - 非调仓日：沿用上一次调仓权重；若尚未调仓则全 0

        参数:
            data: 回测引擎传入的数据（index 为交易日）

        返回:
            pd.DataFrame: 每日权重矩阵
        """
        t0 = time.perf_counter()

        # data is expected to have datetime index; use its range to decide backtest span
        if data is None:
            logger.warning("generate_positions input is None")
            return pd.DataFrame()

        # 这里的 `data` 在本项目里通常只是“交易日历容器”（只用 index），
        # 即使 data 没有任何列（columns 为空），也不应该被视为无效输入。
        if len(data.index) == 0:
            logger.warning("generate_positions empty index")
            return pd.DataFrame()

        idx = pd.to_datetime(data.index)
        start = idx.min().date().isoformat()
        end = idx.max().date().isoformat()
        logger.info("generate_positions start | start=%s end=%s n_days=%s", start, end, len(idx))

        # 取数需要覆盖回测起始日前的 lookback 窗口，否则在回测初期（如 2024-01-01 附近）
        # 调仓/计算窗口会因为历史不足而无法产出结果。
        # 这里用 max(data_window*2, min_history_days*2) 的日历日跨度做保守扩展，
        # 并在后续仍以 tail(data_window) 控制实际使用的交易日数量。
        lookback_days = int(max(self.cfg.data_window, self.cfg.min_history_days) * 2)
        fetch_start = (idx.min() - pd.Timedelta(days=lookback_days)).date().isoformat()

        close_df = self._fetch_close_matrix(fetch_start, end)

        if close_df.empty:
            logger.warning("generate_positions no close data | elapsed=%.3fs", (time.perf_counter() - t0))
            return pd.DataFrame(index=idx)

        # 注意：这里不能 reindex 到 idx，否则会把 fetch_start 之前的历史数据裁剪掉，
        # 导致回测期初调仓窗口仍然缺历史。
        # 正确做法是保留完整 close_df 作为“历史库”，仅在输出/迭代时使用 idx。
        close_df = close_df.sort_index().ffill()

        # prepare output weights df（必须对齐到原始 idx，即使 close_df 包含更早历史）
        weights_df = pd.DataFrame(index=idx, columns=close_df.columns, dtype=float)

        last_w: Optional[pd.Series] = None
        rebalance_count = 0

        for i, dt in enumerate(idx):
            # weekday: Monday=0 .. Sunday=6
            is_rebalance = (dt.weekday() + 1) == self.cfg.rebalance_weekday

            if not is_rebalance:
                # 非调仓日：持仓沿用
                if last_w is not None:
                    weights_df.iloc[i] = last_w.values
                else:
                    weights_df.iloc[i] = 0.0
                continue

            rebalance_count += 1
            if rebalance_count == 1 or (rebalance_count % 10) == 0:
                logger.info("rebalance tick | dt=%s idx=%s/%s", dt.date().isoformat(), i + 1, len(close_df.index))

            # 调仓日：取最近窗口数据。这里额外取 data_window*2 的日历跨度，
            # 再 tail(data_window) 以尽量保证有足够交易日数据。
            window_start = dt - pd.Timedelta(days=int(self.cfg.data_window * 2))
            px = close_df.loc[window_start:dt].tail(int(self.cfg.data_window))

            if px.shape[0] < int(self.cfg.min_history_days):
                logger.warning(
                    "rebalance window insufficient | dt=%s have=%s require_min_history_days=%s (fetch_start=%s)",
                    dt.date().isoformat(),
                    px.shape[0],
                    int(self.cfg.min_history_days),
                    fetch_start,
                )

            # 过滤“新上市/历史太短”的资产：要求在当前窗口 px 内至少有 min_history_days 个有效价格点
            # - px 的列是资产（ETF），行是交易日
            # - px[c].dropna().shape[0]：统计该资产在窗口内“非空价格”的数量（有效历史长度）
            # - 若有效历史 < min_history_days：说明可能刚上市、停牌缺数据、或拉取数据不全
            #   这类资产会让收益率/协方差估计不稳定，进而影响 EPO 优化结果，因此剔除
            valid_cols = [c for c in px.columns if px[c].dropna().shape[0] >= int(self.cfg.min_history_days)]

            # dropped：本次调仓窗口中被剔除的资产数量（用于日志观测池子被“过滤”得有多厉害）
            dropped = int(px.shape[1] - len(valid_cols))

            # 将窗口价格矩阵裁剪到“历史长度足够”的资产集合；后续优化只在这些资产上进行
            px = px[valid_cols]
            # 取窗口内最后 opt_tail_days 个交易日（把 30 抽成可配置参数）
            px = px.tail(int(self.cfg.opt_tail_days))

            if dropped > 0:
                logger.info("rebalance filter | dt=%s dropped=%s kept=%s", dt.date().isoformat(), dropped, px.shape[1])

            wdict = self._run_optimization(px)
            if not wdict:
                # 若优化失败（数据不足等），当期权重全 0
                logger.warning("rebalance optimization empty | dt=%s", dt.date().isoformat())
                new_w = pd.Series(0.0, index=close_df.columns)
            else:
                # 对齐到完整 ETF 列集合，未参与优化的资产权重填 0
                new_w = pd.Series(wdict)
                new_w = new_w.reindex(close_df.columns).fillna(0.0)

            # 调仓过滤：若新旧权重变化太小，则跳过本次调仓（沿用 last_w）
            if last_w is not None:
                min_delta = float(self.cfg.rebalance_min_delta)
                min_delta = max(0.0, min_delta)
                delta = float(np.abs(new_w.values - last_w.values).sum())
                if delta < min_delta:
                    new_w = last_w

            # 权重平滑：w = (1-beta)*old + beta*new
            if last_w is not None:
                beta = float(self.cfg.rebalance_smoothing_beta)
                beta = float(np.clip(beta, 0.0, 1.0))
                if beta < 1.0:
                    new_w = (1.0 - beta) * last_w + beta * new_w

            last_w = new_w
            weights_df.iloc[i] = last_w.values

        logger.info(
            "generate_positions done | n_assets=%s n_rebalances=%s elapsed=%.3fs",
            close_df.shape[1],
            rebalance_count,
            (time.perf_counter() - t0),
        )

        return weights_df.fillna(0.0)
