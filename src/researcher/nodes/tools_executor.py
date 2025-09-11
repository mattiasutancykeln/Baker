from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import MessagesState
from langchain_core.tools import StructuredTool


class GeneratedToolsNode:
    def __init__(self, *, tools: list[StructuredTool]):
        # Map by tool name for quick lookup
        self._tools = {t.name: t for t in tools}

    def __call__(self, state: MessagesState) -> dict:
        last: AnyMessage = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        out: list[AnyMessage] = []
        for tc in last.tool_calls:
            name = tc.get("name") or ""
            args = tc.get("args") or {}
            tool = self._tools.get(name)
            if not tool:
                content = f"Error: tool '{name}' not found."
            else:
                try:
                    content = tool.invoke(args)  # StructuredTool handles validation
                except Exception as e:  # noqa: BLE001
                    content = f"Error invoking tool '{name}': {str(e)}"
            out.append(ToolMessage(tool_call_id=tc.get("id"), content=str(content), name=name or "unknown"))
        return {"messages": out}


