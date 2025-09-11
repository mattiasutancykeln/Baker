from __future__ import annotations

from typing import Annotated, TypedDict, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    formatter_task: Dict[str, Any]
    colleague_task: Dict[str, Any]


