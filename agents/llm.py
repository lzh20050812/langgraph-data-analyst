"""
LLM 调用抽象层 —— 所有 Agent 通过此模块调用 LLM，不直接硬编码模型名。

设计目的（支持多模型基准对比）：
- 换模型只需改配置或传参，不改 Agent 代码
- 支持 DeepSeek-V3（默认）、GPT-4o-mini 等 OpenAI 兼容 API
- 统一处理 temperature、max_tokens、重试等参数
"""

from functools import lru_cache
from time import perf_counter

from openai import OpenAI
from config.settings import get_settings
from agents.run_context import current_budget, emit_event
from agents.runtime_policy import WorkflowBudgetExceeded


@lru_cache(maxsize=4)
def _cached_client(
    api_key: str, base_url: str, timeout: float, max_retries: int
) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_llm_client() -> OpenAI:
    """获取 OpenAI 兼容的 LLM 客户端。"""
    settings = get_settings()
    return _cached_client(
        settings.LLM_API_KEY,
        settings.LLM_API_BASE,
        settings.LLM_TIMEOUT_SECONDS,
        settings.LLM_MAX_RETRIES,
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

    primary_model = model or settings.LLM_MODEL

    def invoke(selected_model: str):
        budget = current_budget()
        if budget is not None:
            budget.before_llm_call(messages)
        emit_event({"type": "llm_call_started", "model": selected_model})
        started = perf_counter()
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=(
                    temperature
                    if temperature is not None
                    else settings.LLM_TEMPERATURE
                ),
                max_tokens=max_tokens,
            )
        except Exception as exc:
            emit_event({
                "type": "llm_call_failed",
                "model": selected_model,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "error": type(exc).__name__,
            })
            raise
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        event = {
            "type": "llm_call_completed",
            "model": selected_model,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_call_cost_usd": round((
                int(prompt_tokens or 0) * settings.LLM_INPUT_COST_PER_MILLION
                + int(completion_tokens or 0) * settings.LLM_OUTPUT_COST_PER_MILLION
            ) / 1_000_000, 8),
        }
        if budget is not None:
            try:
                event["task_budget"] = budget.record_usage(
                    prompt_tokens, completion_tokens, total_tokens
                )
            finally:
                emit_event(event)
        else:
            emit_event(event)
        return response

    try:
        response = invoke(primary_model)
    except WorkflowBudgetExceeded:
        raise
    except Exception:
        fallback_model = getattr(settings, "LLM_FALLBACK_MODEL", "")
        if not fallback_model or fallback_model == primary_model:
            raise
        emit_event({
            "type": "llm_fallback_started",
            "primary_model": primary_model,
            "fallback_model": fallback_model,
        })
        response = invoke(fallback_model)

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
    # Copy caller-owned messages before adding format instructions.
    formatted_messages = [dict(message) for message in messages]
    if formatted_messages and formatted_messages[0]["role"] == "system":
        formatted_messages[0]["content"] += "\n\n你必须以有效的 JSON 格式返回结果，不要包含其他文字。"
    else:
        formatted_messages.insert(0, {
            "role": "system",
            "content": "你必须以有效的 JSON 格式返回结果，不要包含其他文字。",
        })

    return chat(
        formatted_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
