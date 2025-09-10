from __future__ import annotations

from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.researcher.nodes.pandas_tool_node import PandasToolNode
from src.researcher.tools.pandastool import pandastool as _pandastool
from src.researcher.db import ArrowDatabase


class DFState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], MessagesState]


def _ensure_rules(project_dir: str) -> str:
    path = Path(project_dir) / ".format_rules.md"
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return path.read_text(encoding="utf-8")


def build_data_formatter_app(*, project_dir: str, socket_dir: str, db: ArrowDatabase, model_name: str) -> Any:
    rules = _ensure_rules(project_dir)

    # LLM tool schemas (LLM-facing only)
    def _llm_pandastool(code: str) -> str:  # noqa: ANN001
        return ""

    def _llm_update_descr(name: str, description: str) -> str:  # noqa: ANN001
        return ""

    t_pandas = StructuredTool.from_function(
        _llm_pandastool,
        name="pandastool",
        description=(
            "Execute concise pandas code in the sandbox."
            " Use load(name) and save(df, name) to read/write datasets in the project database."
        ),
    )
    t_update = StructuredTool.from_function(
        _llm_update_descr,
        name="update_data_description",
        description=(
            "Update natural-language description for a dataset (columns, rows, expectations)."
        ),
    )

    model = init_chat_model(model=model_name)
    llm = model.bind_tools([t_pandas, t_update])

    def step(state: MessagesState) -> dict:
        sys = SystemMessage(content=(
            "You are the Data Formatter. You exclusively manage data formatting and storage for this project.\n"
            "- Single source of truth: project database at data_dir. Use dataset-name keys.\n"
            "- Minimize dataset proliferation; when safe, merge to reduce duplicates; preserve uniqueness.\n"
            "- Maintain 'descr' entries: after significant changes or new datasets, call update_data_description.\n"
            "- Never print large data; summarize actions and results.\n"
            "- Consult format rules below each turn.\n\n"
            f"Format Rules:\n{rules}"
        ))
        ai: AIMessage = llm.invoke([sys, *state["messages"]])
        return {"messages": [ai]}

    graph = StateGraph(MessagesState)
    graph.add_node("step", step)
    graph.add_node(
        "tool",
        PandasToolNode(project_dir=project_dir, socket_dir=socket_dir, db=db, cfg={}),
    )
    graph.add_edge(START, "step")

    def need_tool(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool"
        return "end"

    graph.add_conditional_edges("step", need_tool, {"tool": "tool", "end": END})
    graph.add_edge("tool", "step")

    memory_dir = Path(project_dir) / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver.from_conn_string(str(memory_dir / "checkpoints.sqlite"))
    return graph.compile(checkpointer=saver)


