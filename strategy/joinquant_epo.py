from jqdata import *
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.covariance import LedoitWolf, OAS
from arch import arch_model
from scipy.stats import linregress

#初始化函数 
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.002))
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0002, close_commission=0.0002, close_today_commission=0, min_commission=5), type='fund')
    log.set_level('order', 'error')
    g.etf_pool = ['159985.XSHE',  #豆粕ETF
                    '518800.XSHG', #黄金基金ETF
                    '159915.XSHE', #创业板ETF
                    '513100.XSHG', #纳指ETF
                    '513600.XSHG', #恒指ETF
                    '160216.XSHE', #国泰商品LOF
                    '159930.XSHE', #能源ETF
                    '159980.XSHE', #有色ETF
                    ]

    run_weekly(trade, 3, '11:15', '000001.XSHE', False)

def dynamic_w_ledoitwolf(returns):
    lw = LedoitWolf().fit(returns.values)
    # LedoitWolf 给出的是 shrinkage intensity towards scaled identity,
    # 具体如何映射到你代码的 w 需小心：
    # 假设 lw.shrinkage_ 在 [0,1] 且表示收缩强度，直接返回即可
    return float(lw.shrinkage_)
    
def garch_vol_forecast(series):
    # 需要去掉 NaN
    series = series.dropna()
    am = arch_model(series, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    # 返回下一期预测的方差
    fcast = res.forecast(horizon=1).variance.values[-1][0]
    return np.sqrt(fcast)

def dynamic_anchor_garch(returns):
    vols = []
    for c in returns.columns:
        vols.append(garch_vol_forecast(returns[c]))
    vols = np.array(vols)

    # Risk-Parity anchor
    inv_vol = 1/vols
    anchor = inv_vol / inv_vol.sum()

    return anchor

# 工具函数：rank 标准化
def rank_normalize(x):
    """
    将序列 rank 到 [-1, 1]
    """
    r = x.rank()
    r = (r - r.mean()) / (r.max() - r.min())
    return r.fillna(0)


# 唯一 signal：趋势强度（不加任何因子）
def ts_trend_only_signal(prices, window=120):
    signal = {}

    for col in prices.columns:
        y = np.log(prices[col].iloc[-window:])
        x = np.arange(len(y))

        slope, _, r_value, _, _ = linregress(x, y)

        # 只做一件事：趋势存在就给信号
        s = max(0, slope)
        signal[col] = s

    return np.array(list(signal.values()))


# 时间衰减 Sharpe
def ts_sharpe_signal(prices, span=60):
    returns = prices.pct_change().dropna()
    mean = returns.ewm(span=span).mean().iloc[-1]
    std = returns.ewm(span=span).std().iloc[-1]
    return (mean / std).values
    

def dynamic_lambda(port_vol_annual):
    if port_vol_annual > 0.18:
        return 3.0
    elif port_vol_annual > 0.14:
        return 2.5
    else:
        return 2.0


def epo(x, signal, lambda_, method="simple", w=None, anchor=None, normalize=True, endogenous=True):
    n = x.shape[1]
    vcov = x.cov()
    corr = x.corr()
    I = np.eye(n)
    V = np.diag(np.diag(vcov))
    std = np.sqrt(V)
    s = signal
    a = anchor

    shrunk_cor = ((1 - w) * I @ corr.values) + (w * I)  # equation 7
    cov_tilde = std @ shrunk_cor @ std  # topic 2.II: page 11
    shrunk_cov =  (1 - w) * cov_tilde + w * V # equation 15
    inv_shrunk_cov = solve(cov_tilde, I)

    if method == "simple":
        epo = (1 / lambda_) * inv_shrunk_cov @ signal  # equation 16
    elif method == "anchored":
        if endogenous:
            gamma = np.sqrt(a.T @ cov_tilde @ a) / np.sqrt(s.T @ inv_shrunk_cov @ cov_tilde @ inv_shrunk_cov @ s)
            epo = inv_shrunk_cov @ (((1 - w) * gamma * s) + ((w * I @ V @ a)))
        else:
            epo = inv_shrunk_cov @ (((1 - w) * (1 / lambda_) * s) + ((w * I @ V @ a)))
    else:
        raise ValueError("`method` not accepted. Try `simple` or `anchored` instead.")

    if normalize:
        epo = [0 if a < 0 else a for a in epo]
        epo = epo / np.sum(epo)

    return epo, cov_tilde

# 定义获取数据并调用优化函数的函数
def run_optimization(stocks, end_date):
    prices = get_price(stocks, count=30, end_date=end_date, frequency='daily', fields=['close'])['close']
    returns = prices.pct_change().dropna() 
    d = returns.var()
    
    # 用等权组合近 60 日波动作为 proxy
    ew = np.ones(len(returns.columns)) / len(returns.columns)
    port_vol_daily = np.sqrt(ew.T @ returns.cov() @ ew)
    port_vol_annual = port_vol_daily * np.sqrt(252)

    #signal = (1/d) / (1/d).sum()
    #signal = returns.mean() #历史平均收益
    #signal = -returns.std() # 最小波动
    #signal = returns[-60:].mean() #历史收益的动量
    
    signal_std = returns.mean() / returns.std() #风险调整后的收益

    #raw_signal = build_epo_signal(prices, returns)
    #signal_rank = cross_sectional_rank(raw_signal, method="symmetric")
    #加一个「零权重区」避免中性资产频繁进出
    # signal_rank[np.abs(signal_rank) < 0.3] = 0

    #signal = ts_trend_signal(prices)
    #signal = ts_multi_factor_signal(prices, returns)
    # signal = ts_trend_only_signal(prices)
    signal = ts_sharpe_signal(prices, span=60)
    signal = np.tanh(signal)
    
    #对负 signal 轻度惩罚，而不是对正 signal 平滑
    # 更快离场，但不延迟进场。
    # signal[signal < 0] *= 1.5
    
    lambda_ = dynamic_lambda(port_vol_annual)

    #w_dyn = dynamic_w_ledoitwolf(returns)
    w_raw = dynamic_w_ledoitwolf(returns)
    w_dyn = np.clip(w_raw, 0.2, 0.6)
    # 风险平价（Risk Parity）作为 anchor
    # vol = returns.std()
    # anchor = (1/vol) / (1/vol).sum()

    # GARCH作为anchor
    anchor = dynamic_anchor_garch(returns)
    
    # anchor 只在 rank > 0 的资产上生效
    #anchor[signal_rank <= 0] = 0
    #anchor = anchor / anchor.sum()
    raw, shrunk_cov = epo(x = returns, 
                        signal = signal_std, 
                        lambda_ = 2.0, 
                        method = "anchored", 
                        #method = "simple", 
                        w = w_dyn, 
                        anchor = anchor,
                        endogenous = True,
                        normalize=False)
    
    # --- 5. 去掉负权（可选）---
    raw = np.clip(raw, 0, None)

    # 若全部 <= 0，则退回等权
    if raw.sum() == 0:
        weights = np.ones(len(stocks)) / len(stocks)
    else:
        #weights = raw / raw.sum()     # 归一化成资金权重
        weights = np.clip(raw, 0, None)
        if weights.sum() > 0:
            weights = weights / weights.sum()

    # ==================================================
    # 6. 风险目标缩放（方法 A 关键步骤，可关闭）
    # ==================================================
    # ⚠ 如果你想维持资金权重和=1，不要做风险缩放；下面模块可开启可关闭。
    use_risk_target = True
    target_vol_annual = 0.12

    if use_risk_target:
        # 注意：shrunk_cov 是日频协方差
        port_vol_daily = np.sqrt(weights.T @ shrunk_cov @ weights)
        port_vol_annual = port_vol_daily * np.sqrt(252)

        if port_vol_annual > 0:
            scale = target_vol_annual / port_vol_annual
            weights = weights * scale               # 风险调整后的真实仓位
            # 如果希望 sum(weights)=1，可以再归一化（看你的交易逻辑）
            # weights = weights / np.sum(np.abs(weights))

    # --- 7. 转成 dict 返回 ---
    weights =  dict(zip(stocks, weights))
    return weights
    
# 交易
def trade(context):
    end_date = context.previous_date 
    etf_pool = filter_new_stock(context,g.etf_pool,30)
    weights = run_optimization(etf_pool, end_date)
    total_value = context.portfolio.total_value 
    for s in weights.keys():
        value = total_value * weights[s] 
        order_target_value(s, value)
        
def filter_new_stock(context, stock_list, d):
    yesterday = context.previous_date
    return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=d)]