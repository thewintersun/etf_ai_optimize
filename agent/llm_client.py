"""agent/llm_client.py

LLM 调用客户端（桩实现）。

该模块抽象了与 LLM 交互的接口，但目前只包含一个桩实现 `call_llm`。
它的主要作用是根据 `mode` 决定行为：

- `cursor_manual`: 直接返回空结果，因为参数是由人工在 Cursor 中修改的。
- `openai`: 预留接口，需要补充 OpenAI API key 与 client 调用逻辑。

未来可以扩展该模块以支持更多 LLM provider。
"""

import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    """LLM 调用结果的统一数据结构。"""

    text: str


def call_llm(prompt: str, mode: str) -> LLMResult:
    """根据指定模式调用 LLM（或不调用）。

    参数:
        prompt: 输入给 LLM 的指令文本。
        mode: 模式，决定具体行为。
            - `cursor_manual`: 不做任何事，直接返回空结果。
            - `openai`: 检查 API key，但抛出 `NotImplementedError`。

    返回:
        LLMResult 实例。

    异常:
        RuntimeError: 当 `mode=openai` 但 `OPENAI_API_KEY` 未设置时。
        NotImplementedError: 当 `mode=openai` 时（因为尚未实现）。
        ValueError: 当 `mode` 不被识别时。
    """

    if mode == "cursor_manual":
        # 在 manual 模式下，agent 只负责生成 prompt，不负责调用 LLM。
        return LLMResult(text="")

    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # TODO: 补充 openai client 的初始化与调用逻辑。
        raise NotImplementedError("openai mode not implemented in this template")

    raise ValueError(f"Unknown mode: {mode}")
