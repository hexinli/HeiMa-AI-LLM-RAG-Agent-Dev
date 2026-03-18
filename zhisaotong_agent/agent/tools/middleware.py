"""
工具调用与模型调用相关的中间件（监控、日志、动态提示词切换等）。

约束：
- 对外交互保持稳定：函数名、装饰器、入参与返回值类型不变；
- 仅增强健壮性与生产可用性（日志安全、异常保留栈、边界条件处理）。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from zhisaotong_agent.utils.logger_handler import get_logger
from zhisaotong_agent.utils.prompt_loader import load_report_prompts, load_system_prompts

logger = get_logger(__name__)


def _redact_mapping(obj: Mapping[str, Any]) -> dict[str, Any]:
    """
    对常见敏感字段做脱敏，避免把密钥/令牌/隐私直接写入日志。
    仅用于日志展示，不影响工具真实入参。
    """

    sensitive_keys = {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "session",
        "session_id",
        "phone",
        "mobile",
        "email",
        "id_card",
    }

    redacted: dict[str, Any] = {}
    for k, v in obj.items():
        key_lower = str(k).lower()
        if key_lower in sensitive_keys:
            redacted[k] = "***REDACTED***"
        else:
            redacted[k] = v
    return redacted


def _safe_preview(value: Any, *, max_len: int = 2000) -> str:
    """
    将任意对象转为可安全打印的短字符串，防止日志爆炸/序列化异常。
    """

    try:
        if isinstance(value, Mapping):
            value = _redact_mapping(value)  # type: ignore[assignment]
        text = repr(value)
    except Exception:
        text = "<unreprable>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控：记录工具名与入参，并在需要时在 runtime.context 打标记。
    """

    tool_call = getattr(request, "tool_call", None) or {}
    tool_name = tool_call.get("name", "<unknown>")
    tool_args = tool_call.get("args", None)

    # 生产最佳实践：避免直接打印完整参数（可能含敏感信息/超长文本）
    logger.info("[tool monitor]执行工具：%s", tool_name)
    logger.info("[tool monitor]传入参数：%s", _safe_preview(tool_args))

    try:
        result = handler(request)
        logger.info("[tool monitor]工具%s调用成功", tool_name)

        # 保持外部交互一致：仍然以同样 key 打标记
        if tool_name == "fill_context_for_report":
            # Runtime.context 通常是跨节点共享的 dict；这里仅设置布尔标志位
            request.runtime.context["report"] = True

        return result
    except Exception:
        # 使用 exception 记录堆栈，且用 bare raise 保留原始 traceback
        logger.exception("工具%s调用失败", tool_name)
        raise


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """
    在模型执行前输出日志（消息条数与最新一条消息概览）。
    """

    messages = state.get("messages") if isinstance(state, dict) else None
    msg_count = len(messages) if isinstance(messages, list) else 0
    logger.info("[log_before_model]即将调用模型，带有%d条消息。", msg_count)

    if not messages:
        return None

    last = messages[-1]
    try:
        content = getattr(last, "content", None)
        content_text = content.strip() if isinstance(content, str) else _safe_preview(content)
        logger.debug("[log_before_model]%s | %s", type(last).__name__, content_text)
    except Exception:
        logger.debug("[log_before_model]%s | <unloggable message>", type(last).__name__)

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    动态切换提示词：当 runtime.context['report'] 为 True 时切换为报告提示词。
    """

    is_report = bool(request.runtime.context.get("report", False))
    if is_report:
        return load_report_prompts()
    return load_system_prompts()

