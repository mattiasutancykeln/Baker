from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage

from src.researcher.base import UnifiedNodeBase
from src.researcher.tools.registry import LoadedTool


class ColleagueNode(UnifiedNodeBase):
    def __init__(self, *, name: str, system_prompt: str, tools: dict[str, LoadedTool], model: str, model_extra: dict | None = None, **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self._name = name
        self._system_prompt = (
            f"You are the colleague '{name}'.\n" + system_prompt + "\n\n"
            "Guidelines:\n"
            "- Use the bound tools to perform your work.\n"
            "- Save outputs under clear names; summarize results concisely.\n"
            "- Do not print large data; use dataset_head/dataset_describe if needed.\n"
        )
        self._llm = init_chat_model(model=model)
        self._tools = tools

    def __call__(self, state: dict) -> dict:
        model = self.bind_default_tools(self._llm, extra_tools=[t.tool for t in self._tools.values()])
        ai: AIMessage = model.invoke([SystemMessage(content=self._system_prompt), *state.get("messages", [])])
        return {"messages": [ai]}


