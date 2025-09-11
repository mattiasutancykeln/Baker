from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.researcher.db import ArrowDatabase
from src.researcher.nodes.colleague import ColleagueNode
from src.researcher.nodes.tools_executor import GeneratedToolsNode
from src.researcher.tools.registry import LoadedTool
from src.researcher.base import UnifiedNodeBase


def build_colleague_app(
    *,
    project_dir: str,
    db: ArrowDatabase,
    model_name: str,
    name: str,
    system_prompt: str,
    tools: dict[str, LoadedTool],
) -> Any:
    graph = StateGraph(MessagesState)

    node = ColleagueNode(
        name=name,
        system_prompt=system_prompt,
        tools=tools,
        model=model_name,
        db=db,
        cfg=None,
    )
    # Merge colleague tools with base read-only tools
    base_tools = ColleagueNode(name=name, system_prompt=system_prompt, tools={}, model=model_name, db=db, cfg=None).get_default_tools()
    exec_node = GeneratedToolsNode(tools=base_tools + [t.tool for t in tools.values()])

    llm_key = "step"
    tool_key = "tools"

    graph.add_node(llm_key, node)
    graph.add_node(tool_key, exec_node)
    graph.add_edge(START, llm_key)

    def need_tool(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return tool_key
        return END

    graph.add_conditional_edges(llm_key, need_tool, {tool_key: tool_key, END: END})
    graph.add_edge(tool_key, llm_key)

    memory_dir = Path(project_dir) / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver.from_conn_string(str(memory_dir / "checkpoints.sqlite"))
    app = graph.compile(checkpointer=saver)
    return app


