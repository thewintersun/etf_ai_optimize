"""agent/guardrails.py

参数文件护栏（guardrails）。

该模块用于在“人工/LLM 修改策略参数文件”之后做安全校验，避免：
- 参数 key 被新增/删除（导致策略结构变化、或引入“作弊”逻辑）。
- 参数值越界、类型不符合预期。
- 参数文件中出现明确禁止的字符串模式（例如未来函数、读取外部数据等）。

实现上采取两种手段：
- AST 静态解析：提取 `PARAMS = {...}` 的 key 集合。
- 动态执行：对 params 文件做 `exec`，以便拿到运行时的 `PARAMS` dict 并校验值。

注意：动态执行存在潜在风险，因此通常会配合 `forbidden_patterns` 做额外限制。
"""

import ast
from pathlib import Path


def check_forbidden_patterns(text: str, forbidden: list[str]) -> None:
    """检查文本中是否包含被禁止的模式。

    参数:
        text: 参数文件的完整文本。
        forbidden: 禁止出现的字符串列表。

    异常:
        ValueError: 命中任一 forbidden pattern 时抛出。
    """

    for p in forbidden:
        if p in text:
            raise ValueError(f"Forbidden pattern found in strategy_params.py: {p}")


def extract_params_keys(text: str) -> set[str]:
    """从 params 文件文本中提取 `PARAMS` 字典的 key 集合。

    说明：
        - 仅支持形如 `PARAMS = {"a": 1, "b": 2}` 的顶层赋值场景。
        - 如果未找到或结构不符合预期，返回空集合。

    参数:
        text: 参数文件文本。

    返回:
        `PARAMS` 的 key 集合。
    """

    tree = ast.parse(text)
    for node in tree.body:
        # 只关心顶层赋值语句。
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PARAMS":
                    # 只接受字面量 dict，避免复杂表达式导致无法静态提取。
                    if isinstance(node.value, ast.Dict):
                        keys = set()
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys.add(k.value)
                        return keys
    return set()


def check_params_file(
    params_path: str | Path,
    *,
    base_keys: set[str],
    forbid_new_keys: bool,
    params_min: float,
    params_max: float,
    forbidden_patterns: list[str],
) -> None:
    """对策略参数文件做完整护栏校验。

    参数:
        params_path: 参数文件路径（例如 strategy/strategy_params.py）。
        base_keys: 基准 key 集合（来自迭代开始前的 params 文件）。
        forbid_new_keys: 是否禁止新增/删除 key。
        params_min: 允许的最小参数值。
        params_max: 允许的最大参数值。
        forbidden_patterns: 禁止出现的字符串模式。

    异常:
        ValueError: 任一校验不通过时抛出。

    校验内容:
        - forbidden_patterns：文本中不得包含禁用模式。
        - PARAMS keys：若启用 forbid_new_keys，则 key 集合必须与 base_keys 一致。
        - PARAMS 类型：执行后必须得到 dict。
        - PARAMS 值：必须是数值型（不允许 bool），且在 [params_min, params_max] 范围内。
    """

    text = Path(params_path).read_text(encoding="utf-8")

    # 1) 文本级黑名单检查。
    check_forbidden_patterns(text, forbidden_patterns)

    # 2) 静态解析 PARAMS key 集合，约束结构不被改动。
    keys = extract_params_keys(text)
    if forbid_new_keys and keys and base_keys and keys != base_keys:
        missing = base_keys - keys
        added = keys - base_keys
        raise ValueError(f"PARAMS keys changed. missing={sorted(missing)} added={sorted(added)}")

    # 3) 动态执行 params 文件，拿到运行时的 PARAMS。
    ns: dict = {}
    exec(compile(text, str(params_path), "exec"), ns, ns)
    params = ns.get("PARAMS")
    if not isinstance(params, dict):
        raise ValueError("PARAMS must be a dict")

    # 4) 校验每个参数的类型与范围。
    for k, v in params.items():
        # bool 是 int 的子类，这里显式排除，避免把 True/False 当作数值参数。
        if isinstance(v, bool):
            raise ValueError(f"PARAMS[{k}] must not be bool")
        if not isinstance(v, (int, float)):
            raise ValueError(f"PARAMS[{k}] must be numeric")
        fv = float(v)
        if fv < params_min or fv > params_max:
            raise ValueError(f"PARAMS[{k}]={fv} out of range [{params_min}, {params_max}]")
