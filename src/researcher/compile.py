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

    researcher = ResearcherNode(system_prompt=system_prompt, model=model_name)

    # Formatter node: invoke sub-app and pass its last AI message back into the main thread
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
        # Find the last AIMessage with tool_calls to extract the tool_call_id
        tool_call_id = None
        last_msgs = state.get("messages", [])
        if last_msgs:
            last_ai = last_msgs[-1]
            try:
                calls = getattr(last_ai, "tool_calls", None) or []
                if calls:
                    tool_call_id = calls[0].get("id")
            except Exception:
                tool_call_id = None
        content = getattr(msgs[-1], "content", "OK") if msgs else "OK"
        tm = ToolMessage(tool_call_id=tool_call_id or "", content=content, name="delegate_to_formatter")
        return {"messages": [tm]}

    graph.add_node("researcher", researcher)
    graph.add_node("formatter", formatter_node)
    graph.add_edge(START, "researcher")
    # Delegate via Command from researcher → formatter, otherwise end
    graph.add_conditional_edges(
        "researcher",
        lambda s: "formatter" if isinstance(s, Command) else END,
        {"formatter": "formatter", END: END},
    )
    # After formatter completes, return to researcher for summary/follow-up
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


