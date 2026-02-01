"""agent/agent.py

该模块实现一个“人工在回路（human-in-the-loop）”的参数优化循环：

- 每轮迭代先运行一次回测（由 `config.yaml` 指定命令）并读取 metrics 输出。
- 通过 `agent.evaluator.score()` 将 metrics 按权重聚合成一个标量 score。
- 根据目标/约束/当前 params 生成一份给人类（或 LLM）看的 prompt，提示如何调整策略参数。
- 人工在 Cursor 中按 prompt 修改参数文件后，agent 做护栏校验（参数范围、是否新增 key、禁用模式等）。
  - 校验失败会回滚到本轮开始前的参数快照，避免把策略改坏。
- 每轮迭代会在 runs 目录下落盘完整的 before/after 参数、metrics、score、summary，便于追溯。

注意：
- 本模块本身不会自动改参数（目前仅支持 `cursor_manual` 模式）。
- metrics 的生成位置、params 文件位置、runs 输出目录均由 `config.yaml` 的 `paths` 决定。
"""

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from agent.evaluator import score as score_fn
from agent.guardrails import check_params_file, extract_params_keys
from agent.prompt_builder import build_prompt, write_agent_prompt


def _load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置文件。

    参数:
        path: YAML 文件路径。

    返回:
        解析后的 dict。

    约束/假设:
        - 文件必须是 UTF-8 编码。
        - YAML 内容需要能被 `yaml.safe_load` 解析为映射类型。
    """

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _run_cmd(cmd: list[str]) -> None:
    """运行外部命令并在失败时抛出包含 stdout/stderr 的异常。

    这里选择 `capture_output=True` 是为了在失败时把关键信息写入异常，便于定位。

    参数:
        cmd: 形如 `["python", "-m", "backtest.run_backtest"]` 的命令列表。

    异常:
        RuntimeError: 当返回码非 0 时抛出，并附带 stdout/stderr。
    """

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + f"cmd={cmd}\n"
            + f"stdout=\n{p.stdout}\n"
            + f"stderr=\n{p.stderr}\n"
        )


def _read_text(path: str | Path) -> str:
    """以 UTF-8 读取文本文件内容。"""

    return Path(path).read_text(encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    """以 UTF-8 写入文本文件内容（覆盖写）。"""

    Path(path).write_text(text, encoding="utf-8")


def _ensure_dirs(*paths: str | Path) -> None:
    """确保目录存在（等价于 mkdir -p）。"""

    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """主循环入口。

    整体流程：
        1. 读取 `config.yaml`（agent 配置、目标/权重、约束、路径等）。
        2. 初始化 runs 目录与基准参数 key 集合（用于后续“禁止新增 key”的护栏）。
        3. 循环 `iterations` 次：
            - 保存本轮参数快照（before）。
            - 运行回测命令并读取 metrics 输出。
            - 计算 score，并把 metrics/score/summary 等落盘到本轮目录。
            - 生成 prompt，提示如何调整策略参数。
            - 等待人工修改参数文件（cursor_manual）。
            - 运行护栏校验；失败则回滚参数文件并记录错误。
            - 保存本轮参数快照（after）。

    注意：
        - 该函数假设 backtest 命令会生成 `paths.metrics_out` 指向的 JSON 文件。
        - 本函数只负责调度与落盘，不关心策略细节；策略参数合法性由 guardrails 负责。
    """

    # 读取主配置（建议保证字段齐全，否则会触发 KeyError 以暴露配置问题）。
    cfg = _load_yaml("config.yaml")

    # agent 自身的运行模式与迭代参数。
    mode = cfg["agent"]["mode"]
    iterations = int(cfg["agent"]["iterations"])
    sleep_seconds = int(cfg["agent"]["sleep_seconds"])
    backtest_cmd = list(cfg["agent"]["backtest_cmd"])

    # 打分目标与各指标权重。
    target = cfg["objectives"]["target"]
    weights = cfg["objectives"]["weights"]

    # 可选的约束配置：参数范围、禁止新增 key、禁止出现的代码模式等。
    constraints = cfg.get("constraints", {})
    params_range = constraints.get("params_range", {"min": 0.1, "max": 200})
    forbid_new_keys = bool(constraints.get("forbid_new_keys", True))
    forbidden_patterns = list(constraints.get("forbidden_patterns", []))

    # 项目路径配置：参数文件、metrics 输出、每轮 runs 目录。
    paths = cfg["paths"]
    params_path = Path(paths["strategy_params"])
    metrics_out = Path(paths["metrics_out"])
    runs_dir = Path(paths["runs_dir"])

    _ensure_dirs(runs_dir)

    # 记录“基准参数 key 集合”，用于后续护栏校验：是否在 params 文件中新增了变量名。
    base_params_text = _read_text(params_path)
    base_keys = extract_params_keys(base_params_text)

    # 追踪全局最优 score（仅用于汇总展示/落盘）。
    best_score = None
    best_iter = None

    for i in range(iterations):
        # 1. 创建本轮迭代目录，并备份参数文件（before）。
        iter_dir = runs_dir / f"iter_{i:04d}"
        _ensure_dirs(iter_dir)
        shutil.copy2(params_path, iter_dir / "params_before.py")

        # 2. 运行回测命令并读取 metrics 输出。
        _run_cmd(backtest_cmd)
        metrics = json.loads(metrics_out.read_text(encoding="utf-8"))

        # 3. 计算 score 并落盘（metrics.json + score.txt）。
        s = score_fn(metrics, weights)
        (iter_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (iter_dir / "score.txt").write_text(str(s), encoding="utf-8")

        # 4. 更新全局最优 score（仅用于记录，不影响后续逻辑）。
        if best_score is None or s > best_score:
            best_score = s
            best_iter = i

        # 5. 生成 prompt 并落盘（agent_prompt.txt）。
        prompt = build_prompt(
            metrics=metrics,
            score=s,
            target=target,
            constraints=constraints,
            base_params_text=_read_text(params_path),
        )
        write_agent_prompt(iter_dir / "agent_prompt.txt", prompt)
        write_agent_prompt("agent_prompt.txt", prompt)  # 额外写一份到项目根目录，方便查看。

        # 6. 等待人工修改参数文件（当前仅支持 cursor_manual 模式）。
        if mode == "cursor_manual":
            # 你可以在 Cursor 里打开 agent_prompt.txt 和 strategy/strategy_params.py
            # 按 prompt 修改 params，然后等待下一轮
            time.sleep(sleep_seconds)
        else:
            raise NotImplementedError(f"mode not supported yet: {mode}")

        # 7. 护栏校验（防改坏/防作弊）。
        try:
            check_params_file(
                params_path,
                base_keys=base_keys,
                forbid_new_keys=forbid_new_keys,
                params_min=float(params_range.get("min", 0.1)),
                params_max=float(params_range.get("max", 200)),
                forbidden_patterns=forbidden_patterns,
            )
        except Exception as e:
            # 校验失败时回滚到修改前的参数文件，并记录错误信息。
            shutil.copy2(iter_dir / "params_before.py", params_path)
            (iter_dir / "guardrails_error.txt").write_text(str(e), encoding="utf-8")

        # 8. 保存修改后的参数文件（after）和汇总信息。
        shutil.copy2(params_path, iter_dir / "params_after.py")

        summary = {
            "iter": i,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "score": s,
            "best_score": best_score,
            "best_iter": best_iter,
        }
        (iter_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
