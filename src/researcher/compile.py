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
from src.researcher.nodes.researcher_tools_executor import ResearcherToolsExecutor
from src.researcher.db import ArrowDatabase
from src.researcher.project_config import ensure_project_config
from src.researcher.tools.registry import load_generated_tools
from src.researcher.nodes.colleague_app import build_colleague_app


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
        # Allow up to 100 internal graph recursions per user turn by default.
        # This limit resets on every user call to `invoke`.
        recursion_limit = int(os.environ.get("RECURSION_LIMIT", "100"))
        if isinstance(user_input, str):
            payload = {"messages": [HumanMessage(content=user_input)]}
        elif isinstance(user_input, dict):
            payload = user_input
        else:
            payload = {"messages": user_input}

        return self.app.invoke(
            payload,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            },
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
    Build a research agent with researcher, data formatter, and dynamic colleagues.

    - Reads prompts and colleagues from specs
    - Persists conversation state to `<project_path>/.memory/checkpoints.sqlite`
    """
    specs = _read_specs(project_path)
    system_prompt = specs.get("main_system_prompt", "").strip()
    if not system_prompt:
        raise ValueError("Specs missing required key 'main_system_prompt'.")

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
        parts: list[str] = []
        if instruction:
            parts.append("Task from researcher:\n" + instruction)
        if code:
            parts.append("Execute the following pandas code:\n```python\n" + code + "\n```")
        if not parts:
            parts.append("No specific instruction or code provided. List available datasets and summarize their descriptions.")
        prompt = "\n\n".join(parts)

        recursion_limit = int(os.environ.get("RECURSION_LIMIT", "100"))
        res = formatter_app.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": recursion_limit},
        )
        msgs = res.get("messages", [])
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
        return {"messages": [tm], "formatter_task": {}}

    # Researcher tool executor for local read-only tools only
    researcher_tools_exec = ResearcherToolsExecutor(researcher.get_default_tools())
    def researcher_tool_node(state: ConversationState) -> dict:
        return researcher_tools_exec(state)

    graph.add_node("researcher", researcher)
    graph.add_node("researcher_tool", researcher_tool_node)
    graph.add_node("formatter", formatter_node)
    graph.add_edge(START, "researcher")
    # Ensure formatter visually routes back to researcher after a tool response
    graph.add_edge("formatter", "researcher")

    def after_researcher(state: ConversationState) -> str:
        task = state.get("formatter_task") or {}
        if isinstance(task, dict) and (task.get("instruction") or task.get("code")):
            return "formatter"
        col_task = state.get("colleague_task") or {}
        # Route to colleague if a name is present; instruction may be empty
        if isinstance(col_task, dict) and col_task.get("name"):
            return f"col_{col_task['name']}_llm"
        last_msgs = state.get("messages", [])
        last = last_msgs[-1] if last_msgs else None
        calls = getattr(last, "tool_calls", None) if last else []
        return "researcher_tool" if calls else END

    # Defer adding conditional edges from researcher until colleagues are registered below
    graph.add_edge("researcher_tool", "researcher")

    # Dynamically add colleagues based on specs
    tool_dir = specs.get("tool_registry_path") or str(Path(project_path) / "generated_tools")
    registry = load_generated_tools(tool_dir, db=database, cfg=cfg)

    colleague_llm_keys: list[str] = []
    for col in specs.get("colleagues", []):
        name = (col.get("name") or "").strip()
        if not name:
            continue
        tools = {t: registry[t] for t in col.get("tools", []) if t in registry}
        missing = [t for t in col.get("tools", []) if t not in registry]
        if missing:
            print(f"[compile] Warning: colleague '{name}' missing tools: {', '.join(missing)}")

        # Build standalone colleague sub-app
        colleague_app = build_colleague_app(
            project_dir=cfg.project_dir,
            db=database,
            model_name=model_name,
            name=name,
            system_prompt=col.get("system_prompt", ""),
            tools=tools,
        )

        # Wrapper node to invoke colleague sub-app and return ToolMessage
        def _run_colleague(state: ConversationState, *, _colleague: str = name, _app: Any = colleague_app) -> ConversationState:
            args = state.get("colleague_task", {})
            instruction = (args.get("instruction") or "").strip()
            if not instruction:
                instruction = f"No specific instruction provided. Summarize your available tools and propose next steps, {_colleague}."
            recursion_limit = int(os.environ.get("RECURSION_LIMIT", "100"))
            res = _app.invoke(
                {"messages": [HumanMessage(content=instruction)]},
                config={"recursion_limit": recursion_limit},
            )
            msgs = res.get("messages", [])
            # Bind result to the original delegate tool_call
            tool_call_id = None
            for m in reversed(state.get("messages", [])):
                try:
                    calls = getattr(m, "tool_calls", None) or []
                    for c in calls:
                        if c.get("name") == f"delegate_to_{_colleague}":
                            tool_call_id = c.get("id")
                            break
                    if tool_call_id:
                        break
                except Exception:
                    continue
            content = getattr(msgs[-1], "content", "OK") if msgs else "OK"
            tm = ToolMessage(tool_call_id=tool_call_id or "", content=content, name=f"delegate_to_{_colleague}")
            # Clear colleague task
            return {"messages": [tm], "colleague_task": {}}

        llm_key = f"col_{name}_llm"
        graph.add_node(llm_key, _run_colleague)
        colleague_llm_keys.append(llm_key)
        # Visual edge back to researcher after colleague completes
        graph.add_edge(llm_key, "researcher")

        # Register delegate tool on researcher
        researcher.register_colleague_delegate(name)

    # Now connect researcher conditional edges to all possible targets for nice graph visualization
    cond_map: Dict[str, str] = {"researcher_tool": "researcher_tool", "formatter": "formatter", END: END}
    for key in colleague_llm_keys:
        cond_map[key] = key
    graph.add_conditional_edges("researcher", after_researcher, cond_map)

    app = graph.compile(checkpointer=saver)

    return CompiledAgent(app=app, saver=saver, specs=specs, project_path=project_path)


def create_app() -> Any:
    """
    Zero-argument factory used by `langgraph dev` to serve the research agent
    for the default project `projects/researcher_20250910_190100`.
    """
    project_path = str(Path(__file__).resolve().parents[2] / "projects" / "researcher_20250910_190100")
    compiled = compile_agent(project_path)
    return compiled.app


