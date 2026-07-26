"""
LLM 调用抽象层 —— 所有 Agent 通过此模块调用 LLM，不直接硬编码模型名。

设计目的（对应论文"多模型对比实验"需求）：
- 换模型只需改配置或传参，不改 Agent 代码
- 支持 DeepSeek-V3（默认）、GPT-4o-mini 等 OpenAI 兼容 API
- 统一处理 temperature、max_tokens、重试等参数
"""

from openai import OpenAI
from config.settings import get_settings


def get_llm_client() -> OpenAI:
    """获取 OpenAI 兼容的 LLM 客户端。"""
    settings = get_settings()
    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_BASE,
    )


def chat(
    messages: list[dict],
    model: str = None,
    temperature: float = None,
    max_tokens: int = 2048,
) -> str:
    """
    调用 LLM 完成一次对话，返回文本响应。

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
        model: 模型名（默认从配置读取）
        temperature: 温度（默认从配置读取）
        max_tokens: 最大输出 token 数

    Returns:
        LLM 的文本响应
    """
    settings = get_settings()
    client = get_llm_client()

    response = client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def chat_with_json_output(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    调用 LLM 并要求返回 JSON 格式（用于结构化输出场景，如 Schema Agent 的字段映射）。

    注意：DeepSeek 不完全支持 response_format，这里用 prompt 引导 + 后处理。
    """
    # 在 system prompt 末尾追加 JSON 格式要求
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n你必须以有效的 JSON 格式返回结果，不要包含其他文字。"
    else:
        messages.insert(0, {
            "role": "system",
            "content": "你必须以有效的 JSON 格式返回结果，不要包含其他文字。",
        })

    return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
