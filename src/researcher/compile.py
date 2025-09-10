from __future__ import annotations

import json
import os
from dataclasses import dataclass
import asyncio
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage, ToolMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
 
from src.researcher.states import ConversationState
from src.researcher.nodes.researcher import ResearcherNode
from src.researcher.nodes.data_formatter import build_data_formatter_app
from src.researcher.nodes.pandas_tool_node import PandasToolNode
from src.researcher.db import ArrowDatabase
from src.researcher.project_config import ensure_project_config


@dataclass
class CompiledAgent:
    """Thin wrapper around a compiled LangGraph app with persistent memory.

    Use `.invoke(user_input, thread_id=...)` to run a turn with a given session/thread.
    The same `thread_id` will resume the conversation using the SQLite checkpoint.
    """

    app: Any
    saver: SqliteSaver
    specs: Dict[str, Any]
    project_path: str

    def invoke(self, user_input: Any, *, thread_id: str) -> Any:
        if isinstance(user_input, str):
            payload = {"messages": [HumanMessage(content=user_input)]}
        elif isinstance(user_input, dict):
            payload = user_input
        else:
            payload = {"messages": user_input}

        return self.app.invoke(
            payload,
            config={"configurable": {"thread_id": thread_id}},
        )


def _read_specs(project_path: str) -> Dict[str, Any]:
    specs_path = Path(project_path) / "agent_specs.json"
    if not specs_path.exists():
        raise FileNotFoundError(f"Specs file not found at: {specs_path}")
    with specs_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_memory(project_path: str) -> SqliteSaver:
    memory_dir = Path(project_path) / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    db_path = memory_dir / "checkpoints.sqlite"
    return SqliteSaver.from_conn_string(str(db_path))


def compile_agent(project_path: str) -> CompiledAgent:
    """
    Build a minimal research agent with a single `researcher` node and persistent memory.

    - Reads the system prompt from `<project_path>/agent_specs.json` under key `main_system_prompt`.
    - Persists conversation state to `<project_path>/.memory/checkpoints.sqlite`.

    Returns a `CompiledAgent` which wraps the compiled graph and the saver.
    """
    specs = _read_specs(project_path)
    system_prompt = specs.get("main_system_prompt", "").strip()
    if not system_prompt:
        raise ValueError("Specs missing required key 'main_system_prompt'.")

    # Model: default to Sonnet-4 per user; can be overridden via env if needed
    model_name = os.environ.get("ANTHROPIC_MODEL", "anthropic:claude-sonnet-4-20250514")

    # Ensure config and unified data dir
    cfg = ensure_project_config(project_path)
    database = ArrowDatabase(root_dir=Path(cfg.data_dir))

    saver = _ensure_memory(project_path)

    graph = StateGraph(ConversationState)
    formatter_app = build_data_formatter_app(project_dir=cfg.project_dir, socket_dir=cfg.socket_dir, db=database, model_name=model_name)

    researcher = ResearcherNode(system_prompt=system_prompt, model=model_name, db=database, cfg=cfg)

    # Wrap the formatter sub-app
    def formatter_node(state: ConversationState) -> ConversationState:
        args = state.get("formatter_task", {})
        instruction = (args.get("instruction") or "").strip()
        code = (args.get("code") or args.get("pandas") or "").strip()
        # Build a guaranteed non-empty prompt for the formatter
        parts: list[str] = []
        if instruction:
            parts.append("Task from researcher:\n" + instruction)
        if code:
            parts.append("Execute the following pandas code:\n```python\n" + code + "\n```")
        if not parts:
            parts.append("No specific instruction or code provided. List available datasets and summarize their descriptions.")
        prompt = "\n\n".join(parts)

        res = formatter_app.invoke({"messages": [HumanMessage(content=prompt)]})
        msgs = res.get("messages", [])
        # Build a ToolMessage replying to the researcher's delegate tool call
        # Find the most recent AI message with tool_calls to extract tool_call_id
        tool_call_id = None
        for m in reversed(state.get("messages", [])):
            try:
                calls = getattr(m, "tool_calls", None) or []
                if calls:
                    tool_call_id = calls[0].get("id")
                    break
            except Exception:
                continue
        content = getattr(msgs[-1], "content", "OK") if msgs else "OK"
        tm = ToolMessage(tool_call_id=tool_call_id or "", content=content, name="delegate_to_formatter")
        # Clear delegation payload to avoid repeated routing on subsequent turns
        return {"messages": [tm], "formatter_task": {}}

    # Researcher subgraph: llm step then unified tool node loop with bubble-out
    def researcher_tool_node(state: ConversationState) -> dict:
        # Execute any tool_calls from last AI message via unified node
        tool = PandasToolNode(project_dir=cfg.project_dir, socket_dir=cfg.socket_dir, db=database, cfg={})
        return tool(state)  # returns {"messages": [ToolMessage, ...]}

    graph.add_node("researcher", researcher)
    graph.add_node("researcher_tool", researcher_tool_node)
    graph.add_node("formatter", formatter_node)
    graph.add_edge(START, "researcher")

    def after_researcher(state: ConversationState) -> str:
        # Bubble out only if a non-empty delegation payload is present
        task = state.get("formatter_task") or {}
        if isinstance(task, dict) and (task.get("instruction") or task.get("code")):
            return "formatter"
        last_msgs = state.get("messages", [])
        last = last_msgs[-1] if last_msgs else None
        calls = getattr(last, "tool_calls", None) if last else []
        # If the last AI has tool_calls, run tools; else end the subgraph turn
        return "researcher_tool" if calls else END

    graph.add_conditional_edges("researcher", after_researcher, {"researcher_tool": "researcher_tool", "formatter": "formatter", END: END})
    graph.add_edge("researcher_tool", "researcher")
    # After formatter responds, return to researcher once to summarize, then natural end if no further tools
    graph.add_edge("formatter", "researcher")

    app = graph.compile(checkpointer=saver)

    return CompiledAgent(app=app, saver=saver, specs=specs, project_path=project_path)


def create_app() -> Any:
    """
    Zero-argument factory used by `langgraph dev` to serve the research agent
    for the default project `projects/researcher_20250908_093746`.
    """
    project_path = str(Path(__file__).resolve().parents[2] / "projects" / "researcher_20250908_093746")
    compiled = compile_agent(project_path)
    return compiled.app


