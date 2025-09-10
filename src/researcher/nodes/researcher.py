from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.types import Command

from src.researcher.states import ConversationState


class DelegateFormatterArgs(BaseModel):
    """Delegate a data task to the formatter agent."""
    pandas: str = Field(description="Minimal pandas code to run in the formatter's sandbox", default="")
    instruction: str = Field(description="High-level instruction for the formatter", default="")


class ResearcherNode:
    """
    Minimal reasoning node for multi-turn dialog.

    - Uses Anthropic Sonnet model for reasoning.
    - Injects a system message provided at construction time for every turn.
    - Returns an AI message to be appended to the conversation state.
    """

    SYSTEM_PROMPT_START = """
    You are an objective, helpful research assistant. Communicate concisely and focus on essential information.
    If crucial details are missing or ambiguous, ask 1–2 crisp clarification questions before proceeding.
    """
    SYSTEM_PROMPT_END = """
    Delegation: Use the Data Formatter via the bound delegate tool for any data write/merge/delete and for dataset creation or consolidation. Keep few datasets and ensure uniqueness.
    - Prefer outsourcing complete tasks in a single comprehensive delegate call rather than many small calls.
    - Only include exact pandas code when 100% certain about dataset names and availability; otherwise provide high-level instructions and let the formatter choose safe operations.
    """

    def __init__(self, *, system_prompt: str, model: str = "anthropic:claude-sonnet-4-20250514") -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        self._system_prompt = self.SYSTEM_PROMPT_START + system_prompt + self.SYSTEM_PROMPT_END
        self._llm = init_chat_model(model=model)
        # Bind a delegate tool that signals a graph transition to the formatter node
        def delegate_to_formatter(instruction: str, code: str = "") -> str:  # noqa: ANN001 - schema only; not executed
            return "delegated"
        self._delegate_tool = StructuredTool.from_function(
            delegate_to_formatter,
            name="delegate_to_formatter",
            description=(
                "Delegate any data storage/merge/delete/creation task to the formatter agent. "
                "Optionally include concise pandas code to run in the sandbox."
            ),
        )

    def __call__(self, state: ConversationState) -> dict:
        system = SystemMessage(content=self._system_prompt)
        model = self._llm.bind_tools([self._delegate_tool])
        ai: AIMessage = model.invoke([system, *state["messages"]])  # type: ignore[arg-type]
        if getattr(ai, "tool_calls", None):
            call = ai.tool_calls[0]
            raw_args = call.get("args") or {}
            # Normalize keys for downstream formatter node
            instruction = (raw_args.get("instruction") or "").strip()
            code = (raw_args.get("code") or raw_args.get("pandas") or "").strip()
            return Command(
                goto="formatter",
                update={
                    "messages": [ai],
                    "formatter_task": {"instruction": instruction, "code": code},
                },
            )
        return {"messages": [ai]}


