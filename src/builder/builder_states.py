from typing import Dict, List, Optional, Sequence
from typing_extensions import Annotated, Literal, TypedDict, operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====
class DynamicAgentInputState(TypedDict):
	"""Input state containing user's initial research request"""
	messages: Annotated[Sequence[BaseMessage], add_messages]
	project_dir: Optional[str]

class DynamicAgentState(TypedDict):
	"""Main state for the dynamic research agent workflow"""
	messages: Annotated[Sequence[BaseMessage], add_messages]
	research_requirements: Optional[str]
	colleagues: List[Dict]
	tool_implementations: List[str]
	completed_tool: Annotated[List[Dict], operator.add]
	main_system_prompt: Optional[str]
	final_agent_spec: Optional[str]
	project_dir: Optional[str]
	# Review loop controls and artifacts
	graph_review_iterations: Optional[int]
	max_graph_review_iters: Optional[int]
	compatibility_schema: Optional[str]
	compatibility_schema_path: Optional[str]
	# Optional grouping of tool implementations (integration-critical batches)
	grouped_tool_implementations: Optional[List[List[str]]]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ColleagueState(BaseModel):
	"""Schema for research colleague specification"""
	name: str = Field(description="Unique identifier for the colleague")
	role: str = Field(description="Role title")
	expertise: str = Field(description="Detailed description of expertise area")
	system_prompt: str = Field(description="System prompt defining the colleague's behavior")
	tools: List[str] = Field(description="List of tool names this colleague will use")

	class Config:
		extra = "forbid"

class GraphPlan(BaseModel):
	"""Schema for complete research agent plan"""
	main_system_prompt: str = Field(description="System prompt for main reasoning orchestrator")
	colleagues: List[ColleagueState] = Field(description="List of research colleague specifications")
	tool_implementations: List[str] = Field(description="Implementation commands for ToolBuilders. Always starts with 'Implement [tool_name]...'")

	class Config:
		extra = "forbid"

class ToolImplementation(BaseModel):
	"""Schema for tool implementation output"""
	tool_name: str = Field(description="Name of the implemented tool")
	tool_code: str = Field(description="Complete Python code for the tool")
	tool_description: str = Field(description="Description of what the tool does")

	class Config:
		extra = "forbid"

class ToolBuildState(TypedDict):
	"""State for tool building containing implementation requirements"""
	# Tool implementation inputs (list form; single tool is len==1)
	implementation_commands: Optional[List[str]]
	tool_names: Optional[List[str]]
	project_dir: Optional[str]
	# Reviewer loop tracking
	review_iterations: Optional[int]
	max_review_iters: Optional[int]
	approved: Optional[bool]
	limit_reached: Optional[bool]
	# Latest build artifacts (single or batch)
	# List of tool dicts {tool_name, tool_code, tool_description, tool_card}
	tools: Optional[List[Dict]]
	# Accumulated attempts with scores per tool
	attempts: Optional[Dict]
	# Final tool specification(s)
	completed_tool: Optional[Dict]
	compatibility_schema: Optional[str]
	compatibility_schema_path: Optional[str]

class GraphReviewVerdict(BaseModel):
	"""Reviewer verdict for the graph plan"""
	approved: bool
	critique: str
	required_changes: str
	revised_tool_implementations: List[str] = Field(description="Revised implementation commands when changes are needed")

	class Config:
		extra = "forbid"
