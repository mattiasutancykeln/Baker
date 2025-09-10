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

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ColleagueState(BaseModel):
	"""Schema for research colleague specification"""
	name: str = Field(description="Unique identifier for the colleague")
	role: str = Field(description="Role title")
	expertise: str = Field(description="Detailed description of expertise area")
	system_prompt: str = Field(description="System prompt defining the colleague's behavior")
	tools: List[str] = Field(description="List of tool names this colleague will use")

class GraphPlan(BaseModel):
	"""Schema for complete research agent plan"""
	main_system_prompt: str = Field(description="System prompt for main reasoning orchestrator")
	colleagues: List[ColleagueState] = Field(description="List of research colleague specifications")
	tool_implementations: List[str] = Field(description="Implementation commands for ToolBuilders. Always starts with 'Implement [tool_name]...'")

class ToolImplementation(BaseModel):
	"""Schema for tool implementation output"""
	tool_name: str = Field(description="Name of the implemented tool")
	tool_code: str = Field(description="Complete Python code for the tool")
	tool_description: str = Field(description="Description of what the tool does")

class ToolBuildState(TypedDict):
	"""State for tool building containing implementation requirements"""
	implementation_command: str
	tool_name: str
	project_dir: Optional[str]
	tool_code: Optional[str]
	tool_description: Optional[str]
	tool_card: Optional[str]
	completed_tool: Optional[Dict]  # Final tool specification
