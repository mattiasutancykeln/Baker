from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.types import Command

from src.researcher.states import ConversationState
from src.researcher.base import UnifiedNodeBase


class DelegateFormatterArgs(BaseModel):
    """Delegate a data task to the formatter agent."""
    pandas: str = Field(description="Minimal pandas code to run in the formatter's sandbox", default="")
    instruction: str = Field(description="High-level instruction for the formatter", default="")


class ResearcherNode(UnifiedNodeBase):
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

    def __init__(self, *, system_prompt: str, model: str = "anthropic:claude-sonnet-4-20250514", **base_kwargs) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        super().__init__(**base_kwargs)
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
        model = self.bind_default_tools(self._llm, extra_tools=[self._delegate_tool])
        ai: AIMessage = model.invoke([system, *state["messages"]])  # type: ignore[arg-type]
        if not getattr(ai, "tool_calls", None):
            return {"messages": [ai]}

        # If delegation tool present, bubble out via state for outer routing
        for tc in ai.tool_calls:
            name = tc.get("name")
            if name == "delegate_to_formatter":
                args = tc.get("args") or {}
                instruction = (args.get("instruction") or "").strip()
                code = (args.get("code") or args.get("pandas") or "").strip()
                if instruction or code:
                    return {
                        "messages": [ai],
                        "formatter_task": {"instruction": instruction, "code": code},
                    }
                # If empty args, do not bubble; allow tool loop to run (will be no-op)

        # Otherwise, allow outer graph to route to the unified tool node for local tools
        return {"messages": [ai]}


