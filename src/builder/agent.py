import json
import re
from pathlib import Path

from typing_extensions import Literal, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START
from langgraph.types import Command, Send

from src.builder.builder_states import DynamicAgentInputState, DynamicAgentState
from src.builder.graph_builder import clarify_with_user, graph_builder
from src.builder.tool_builder import create_tool_builder_agent

# ===== CONFIGURATION =====
model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")

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

    # Aggregate all tool implementations
    implemented_tools = {}
    for tool_result in state.get("completed_tool", []):
        implemented_tools[tool_result["name"]] = tool_result["description"]

    # Create final agent specification
    final_spec = {
        "main_system_prompt": state["main_system_prompt"],
        "colleagues": state["colleagues"],
        "implemented_tools": implemented_tools,
        "tool_registry_path": str(tools_dir),
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

def map_tool_implementations(state: DynamicAgentState) -> List[Send]:
    """
    Create Send commands for parallel tool implementation.

    This function is used as a conditional edge that maps each tool
    implementation command to a separate tool_builder instance.
    """

    all_required_tools = {}
    for implementation in state["tool_implementations"]:
        name = re.search(r"Implement\s+(\w+)", implementation).group(1)
        all_required_tools[name] = implementation
    return [
        Send(
            "tool_builder",
            {
                "implementation_command": impl_cmd,
                "tool_name": name,
                "project_dir": state.get("project_dir"),
            },
        )
        for name, impl_cmd in all_required_tools.items()
    ]


def create_research_agent():
    """Create the dynamic research agent workflow with Send API"""

    builder = StateGraph(DynamicAgentState, input_schema=DynamicAgentInputState)

    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("graph_builder", graph_builder)
    builder.add_node("finalize_agent", finalize_agent)
    builder.add_node("tool_builder", create_tool_builder_agent())

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("tool_builder", "finalize_agent")
    builder.add_conditional_edges(
        "graph_builder",
        map_tool_implementations,
        ["finalize_agent"],
    )

    return builder.compile()


# ===== FOR LANGGRAPH SERVER =====
agent = create_research_agent()
