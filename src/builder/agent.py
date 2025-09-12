import json
import re
from pathlib import Path

from typing_extensions import Literal, List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START
from langgraph.types import Command, Send

from src.builder.builder_states import DynamicAgentInputState, DynamicAgentState
from src.builder.graph_builder import clarify_with_user, graph_builder, graph_review, build_compat_schema
from src.builder.tool_builder import create_tool_builder_agent

# ===== CONFIGURATION =====
model = ChatAnthropic(model= "anthropic:claude-sonnet-4-20250514", temperature=0)

def finalize_agent(state: DynamicAgentState) -> Command[Literal["__end__"]]:
    """
    Finalize the agent specification after all tools are built.

    Aggregates all completed tool implementations and creates the
    final agent specification JSON under the project directory.
    """

    project_dir = Path(state.get("project_dir") or "projects/default_researcher")
    project_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = project_dir / "generated_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate all tool implementations (flatten single or batched results)
    implemented_tools = {}
    for tool_result in state.get("completed_tool", []):
        if isinstance(tool_result, list):
            for tr in tool_result:
                implemented_tools[tr.get("name", "")] = tr.get("description", "")
        elif isinstance(tool_result, dict):
            implemented_tools[tool_result.get("name", "")] = tool_result.get("description", "")

    # Create final agent specification
    final_spec = {
        "main_system_prompt": state["main_system_prompt"],
        "colleagues": state["colleagues"],
        "implemented_tools": implemented_tools,
        "tool_registry_path": str(tools_dir),
        "compatibility_schema_path": state.get("compatibility_schema_path"),
    }
    json.dump(final_spec, open(project_dir / "agent_specs.json", "w"))

    return Command(
        goto="__end__",
        update={
            "final_agent_spec": json.dumps(final_spec, indent=2),
            "messages": [
                AIMessage(
                    content=f"Research agent specification complete with {len(implemented_tools)} custom tools!\n\n```json\n{json.dumps(final_spec, indent=2)}\n```"
                )
            ],
        },
    )


# ===== GRAPH CONSTRUCTION =====
def run_tool_builder(state: DynamicAgentState) -> dict:
    """Wrapper to execute the compiled tool_builder subgraph and only bubble up safe keys.

    Avoids concurrent parent-state updates on scalar keys like project_dir.
    """
    app = create_tool_builder_agent()
    child_input = {
        "implementation_commands": state.get("tool_implementations", []),
        "tool_names": [],
        "project_dir": state.get("project_dir"),
        "compatibility_schema": state.get("compatibility_schema"),
        "compatibility_schema_path": state.get("compatibility_schema_path"),
    }
    # If this wrapper is used with map_tool_implementations Sends, those states will
    # already carry the list-form fields. Prefer those when present.
    if state.get("implementation_commands") is not None:
        child_input["implementation_commands"] = state.get("implementation_commands")
    if state.get("tool_names") is not None:
        child_input["tool_names"] = state.get("tool_names")

    res = app.invoke(child_input)
    return {"completed_tool": res.get("completed_tool", [])}


def map_tool_implementations(state: DynamicAgentState) -> List[Send]:
    """
    Create Send commands for parallel tool implementation.

    This function is used as a conditional edge that maps each tool
    implementation command to a separate tool_builder instance.
    """

    # If grouped batches are provided (for tight integration), send batches
    if state.get("grouped_tool_implementations"):
        sends = []
        for batch in state["grouped_tool_implementations"]:
            names = []
            for impl in batch:
                m = re.search(r"Implement\s+(\w+)", impl)
                if m:
                    names.append(m.group(1))
            sends.append(
                Send(
                    "tool_builder",
                    {
                        "implementation_commands": batch,
                        "tool_names": names,
                        "project_dir": state.get("project_dir"),
                        "compatibility_schema": state.get("compatibility_schema"),
                        "compatibility_schema_path": state.get("compatibility_schema_path"),
                    },
                )
            )
        return sends

    # Default: individual tools (send as single-item lists)
    all_required_tools = {}
    for implementation in state.get("tool_implementations", []):
        m = re.search(r"Implement\s+(\w+)", implementation)
        if not m:
            continue
        name = m.group(1)
        all_required_tools[name] = implementation
    return [
        Send(
            "tool_builder",
            {
                "implementation_commands": [impl_cmd],
                "tool_names": [name],
                "project_dir": state.get("project_dir"),
                "compatibility_schema": state.get("compatibility_schema"),
                "compatibility_schema_path": state.get("compatibility_schema_path"),
            },
        )
        for name, impl_cmd in all_required_tools.items()
    ]


def create_research_agent():
    """Create the dynamic research agent workflow with Send API"""

    builder = StateGraph(DynamicAgentState, input_schema=DynamicAgentInputState)

    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("graph_builder", graph_builder)
    builder.add_node("graph_review", graph_review)
    builder.add_node("build_compat_schema", build_compat_schema)
    builder.add_node("finalize_agent", finalize_agent)
    builder.add_node("tool_builder", run_tool_builder)

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("graph_builder", "graph_review")
    # graph_review decides via goto("graph_builder"|"build_compat_schema")
    builder.add_edge("tool_builder", "finalize_agent")
    # After building compatibility schema, fan out to tool builders; if no tools, go finalize
    # Include tool_builder in targets so the visualizer shows the Send path
    builder.add_conditional_edges("build_compat_schema", map_tool_implementations, ["tool_builder", "finalize_agent"])

    return builder.compile()


# ===== FOR LANGGRAPH SERVER =====
agent = create_research_agent()
