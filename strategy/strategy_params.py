"""strategy/strategy_params.py

策略参数文件。

该文件是“人工/LLM 在回路”中唯一允许被修改的文件。
agent 会在每轮迭代后读取该文件，并将其中的 `PARAMS` 字典注入到策略中。
"""

PARAMS = {
    # --- 标的池 ---
    # 聚宽原策略的 ETF 池，注意需要是 data/manager.py 支持的格式
    "start_date": "2025-1-1",
    "end_date": "2025-6-30",

    
    "etf_pool": [
        "159985.SZ",  # 豆粕ETF
        "518800.SH",  # 黄金基金ETF
        "159915.SZ",  # 创业板ETF
        "513100.SH",  # 纳指ETF
        "513600.SH",  # 恒指ETF
        "160216.SZ",  # 国泰商品LOF
        "159930.SZ",  # 能源ETF
        "159980.SZ",  # 有色ETF
    ],

    # 代码到中文名称映射（用于 reports 输出）
    "symbol_name_map": {
        "159985.SZ": "豆粕ETF",
        "518800.SH": "黄金基金ETF",
        "159915.SZ": "创业板ETF",
        "513100.SH": "纳指ETF",
        "513600.SH": "恒指ETF",
        "160216.SZ": "国泰商品LOF",
        "159930.SZ": "能源ETF",
        "159980.SZ": "有色ETF",
        "510300.SH": "沪深300ETF",
    },

    # 基准品种（用于收益曲线对比）
    "benchmark_symbol": "510300.SH",

    # --- 调仓与数据窗口 ---
    "rebalance_weekday": 3,  # 每周几调仓 (1-7, 1=周一)
    "data_window": 90,  # 计算信号与协方差所需的数据窗口（日）
    "min_history_days": 30,  # 上市不足 N 天的 ETF 不参与计算

    # --- EPO 核心参数 ---
    "signal_span": 60,  # 时间衰减 Sharpe 的 span
    "w_clip_min": 0.2,  # 协方差收缩系数 w 的下限
    "w_clip_max": 0.6,  # 协方差收缩系数 w 的上限

    # lambda 控制：anchored 里也会用到（此前代码里写死 2.0，现在已修复）
    "use_dynamic_lambda": True,  # True: 用组合波动自适应 lambda；False: 使用固定 epo_lambda
    "epo_lambda": 2.0,  # 固定 lambda（use_dynamic_lambda=False 时生效）

    # 单一资产权重上限（None 表示不限制）
    "max_weight": None,

    # --- 风险目标（杠杆）控制 ---
    "use_risk_target": True,  # 是否启用风险目标缩放
    "target_vol_annual": 0.14,  # 年化目标波动率（温和抬升收益的起步值）
    "max_leverage": 1.3,  # 杠杆上限（权重总和的上限近似；用于 cap 风险目标缩放的 scale）

    # --- 回测资金与手续费 ---
    "initial_capital": 1_000_000.0,  # 初始资金（元）
    "fee_bps": 2.0,  # 手续费率（基点, bps）。2.0 表示单边万分之二。
    "slippage": 0.001,  # 交易滑点（元/份额）。默认 0.002 元。
}
