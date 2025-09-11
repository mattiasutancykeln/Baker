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

    Only delegate when you can't solve the task through scientific reasoning. Sometimes, a logical argument is enough and an advanced statistical analysis provides less insight.
    """

    def __init__(self, *, system_prompt: str, model: str = "openai:gpt-5-medium", model_extra: dict | None = None, **base_kwargs) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        super().__init__(**base_kwargs)
        self._system_prompt = self.SYSTEM_PROMPT_START + system_prompt + self.SYSTEM_PROMPT_END
        if model_extra:
            self._llm = init_chat_model(model=model, extra_body=model_extra)
        else:
            self._llm = init_chat_model(model=model)
        # Common delegate_to_formatter tool
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
        self._colleague_delegate_tools: list[StructuredTool] = []

    def register_colleague_delegate(self, colleague_name: str) -> None:
        tool_name = f"delegate_to_{colleague_name}"
        def _delegate(instruction: str) -> str:  # noqa: ANN001 - schema only; not executed
            return "delegated"
        tool = StructuredTool.from_function(
            _delegate,
            name=tool_name,
            description=(
                f"Delegate a task to the colleague '{colleague_name}'. Provide a concise instruction."
            ),
        )
        self._colleague_delegate_tools.append(tool)

    def __call__(self, state: ConversationState) -> dict:
        system = SystemMessage(content=self._system_prompt)
        extra = [self._delegate_tool] + self._colleague_delegate_tools
        model = self.bind_default_tools(self._llm, extra_tools=extra)
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
            elif name and name.startswith("delegate_to_"):
                colleague = name[len("delegate_to_"):]
                args = tc.get("args") or {}
                # Accept multiple common keys to be robust
                instruction = (
                    args.get("instruction")
                    or args.get("task")
                    or args.get("input")
                    or args.get("message")
                    or args.get("query")
                    or ""
                )
                instruction = instruction.strip()
                # Always bubble out on any delegate_to_*, even if empty; colleague can request clarification
                return {
                    "messages": [ai],
                    "colleague_task": {"name": colleague, "instruction": instruction},
                }

        # Otherwise, allow outer graph to route to the unified tool node for local tools
        return {"messages": [ai]}


