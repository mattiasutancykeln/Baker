from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import START, END, StateGraph
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.researcher.db import ArrowDatabase
from src.researcher.PandasDBTool import PandasDBTool


@dataclass
class CompiledPandasDemo:
    app: Any


def compile_pandas_demo(project_path: str = "projects/researcher_20250908_093746") -> CompiledPandasDemo:
    """Minimal agent with one node that can call pandas.exec tool."""
    db = ArrowDatabase(root_dir=Path(project_path) / ".arrowdb")
    model_name = os.environ.get("ANTHROPIC_MODEL", "anthropic:claude-sonnet-4-20250514")
    llm = init_chat_model(model=model_name)
    pandas_tool = PandasDBTool(db_dir=Path(project_path) / ".arrowdb")

    def tool_call(name: str, arguments: Dict[str, Any]) -> str:
        if name == "pandas.exec":
            return pandas_tool.run(**arguments)
        return f"[ERROR] Unknown tool: {name}"

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        sys = SystemMessage(content=(
            "You can manage datasets using the 'pandas.exec' tool. "
            "Use concise pandas code; do not print data."
        ))
        messages = state.get("messages", [])
        # NOTE: In a full ReAct setup we'd parse tool calls from model output. Here we just respond.
        ai: AIMessage = llm.invoke([sys, *messages])  # type: ignore
        return {"messages": [ai]}

    graph = StateGraph(dict)
    graph.add_node("demo", node)
    graph.add_edge(START, "demo")
    graph.add_edge("demo", END)
    app = graph.compile()

    # Attach a simple tool entry for tests
    app.pandas_exec = lambda **kw: tool_call("pandas.exec", kw)  # type: ignore[attr-defined]

    return CompiledPandasDemo(app=app)


