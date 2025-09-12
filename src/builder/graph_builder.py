from typing_extensions import Literal, List
import json
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.types import Command, Send
from src.builder.builder_states import DynamicAgentState, GraphPlan, GraphReviewVerdict
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

class ClarificationResponse(BaseModel):
	"""Schema for user clarification decisions"""
	need_clarification: bool = Field(description="Whether more clarification is needed")
	clarifying_question: str = Field(description="Question to ask user for more details")
	research_requirements: str = Field(description="Complete research requirements if no clarification needed")

	class Config:
		extra = "forbid"

model_clarify = init_chat_model("anthropic:claude-sonnet-4-20250514")
model_builder = ChatOpenAI(
	model_name="gpt-5",
	max_tokens=8096,
	reasoning={"effort": "medium"}
)
model_reviewer = ChatOpenAI(
	model_name="gpt-5",
	max_tokens=8096
)

BASE_DIR = Path(__file__).resolve().parent
predefined_tools_path = BASE_DIR / "predefined_tools.json"
PREDEFINED_TOOLS = json.loads(predefined_tools_path.read_text())

# ===== WORKFLOW NODES =====

def clarify_with_user(state: DynamicAgentState) -> Command[Literal["graph_builder", "__end__"]]:
	"""Interactive clarification of research requirements"""
	clarification_prompt = """
	You are helping to design a custom research agent. Review the conversation history and determine
	if you have enough information to design the research team and tools. 
	
	You need to understand:
	1. What type of research is being conducted?
	2. What data sources will be used?
	3. What analyses or outputs are expected?
	4. What level of automation is desired?
	5. Are there any specific methodologies or tools mentioned?
	
	Conversation history:
	{messages}
	
	If you need more information, ask specific clarifying questions.
	If you have enough information, summarize the complete research requirements.
	"""
	
	structured_model = model_clarify.with_structured_output(ClarificationResponse, method="json_schema", strict=True)
	
	response = structured_model.invoke([
		HumanMessage(content=clarification_prompt.format(
			messages=get_buffer_string(state["messages"])
		))
	])
	
	if response.need_clarification:
		return Command(
			goto="__end__",
			update={"messages": [AIMessage(content=response.clarifying_question)]}
		)
	else:
		return Command(
			goto="graph_builder",
			update={
				"research_requirements": response.research_requirements,
				"messages": [AIMessage(content="Perfect! I understand your research needs. Let me design the optimal research team for you.")]
			}
		)


def get_examples():
	examples_path = BASE_DIR / "graph_examples.json"
	with examples_path.open("r") as f:
		examples = json.load(f)

	examples_str = ""
	for n,example in enumerate(examples.keys()):
		examples_str += f"<example_{n}>\n"
		request = examples[example]["request"]
		examples_str += f"<requirements>\n{request}\n</requirements>\n"
		output = examples[example]["output"]
		examples_str += f"<output>\n{output}\n</output>\n"
		examples_str += f"</example_{n}>\n"
	return examples_str


def graph_builder(state: DynamicAgentState) -> Command[Literal["graph_review"]]:
	"""Design the research agent architecture"""
	planning_prompt = """
	You are designing a research agent architecture.
	
	Design the MINIMAL but effective research team following these principles:
	1. Start with just a main reasoning orchestrator
	2. Only add colleagues if they significantly improve performance
	3. Each colleague should have distinct expertise
	4. Minimize tool overlap between colleagues
	5. Prefer using predefined tools when possible
	6. Aim for compatibility by encouraging flexible, general tool signatures and clear descriptions; avoid prescribing strict schemas.
	7. Keep the research agent as simple and general as possible within the scope.

	By default, the research agent will have a reasoning orchestrator and a database manager. These should NOT be added to the colleagues list. 
	- The database manager is completely predefined. You dont needto create a specific colleague for any kind of reading, writing or converting data. 
	- The reasoning orchestrator needs a system prompt with a task, a role, and a list of colleagues. Keep the system_prompt minimal and only include the absolutely necessary information. Don't encourage any specific behaviour, this can significantly reduce performance on generic tasks.
	
	Predefined tools (available to all nodes by default, dont specify): 
	{predefined_tools}
	
	For any tools not in the predefined list, create implementation commands that are clear but general. Make them intuitive for an LLM user who only sees signatures + descriptions.
	Each tool will be implemented in parallel. Ensure user-facing arguments and returns are simple and compatible without enforcing strict formats.
	The research agent stores and reads data from a database automatically passed to tools. Datasets should be referred to by name (strings), not passed as raw data.
	Avoid hardcoding column or schema requirements. If relevant, suggest parameters that allow users to specify columns or options, with sensible defaults in descriptions.
	You will also draft a brief, general compatibility schema to guide all tool builders. This schema is an artifact, not a tool.

	Design guidance (concise):
	- Tools are DB-oriented: all data I/O via the shared db.
	- Feature transforms/encodings within analysis tools are allowed (e.g., scaling, one-hot). General data generation/formatting should be done via the Data Formatter. Keep prep and analysis/modeling as separate tools.
	- When relevant, prefer separate dataset names for training, prediction/evaluation, and outputs.
	
	EXAMPLES:
	{graph_examples}

	Now, design an agent based on the following requirements:
	{research_requirements}
	"""
	
	structured_model = model_builder.with_structured_output(GraphPlan, method="json_schema", strict=True)
	
	response = structured_model.invoke([
		HumanMessage(content=planning_prompt.format(
			research_requirements=state["research_requirements"],
			predefined_tools=json.dumps(PREDEFINED_TOOLS, indent=2),
			graph_examples=get_examples()
		))
	])

	# Compute default project directory if missing
	project_dir = state.get("project_dir") or f"projects/researcher_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

	updated_state = {
		"colleagues": [colleague.dict() for colleague in response.colleagues],
		"tool_implementations": response.tool_implementations or [],
		"main_system_prompt": response.main_system_prompt,
		"project_dir": project_dir,
		"messages": [AIMessage(content="Drafted agent plan. Submitting for compatibility review...")]
	}
	return Command(
		goto="graph_review",
		update=updated_state
	)


def graph_review(state: DynamicAgentState) -> Command[Literal["graph_builder", "build_compat_schema"]]:
	"""Review the agent plan for generality and compatibility, up to 3 iterations"""
	iterations = state.get("graph_review_iterations") or 0
	max_iters = state.get("max_graph_review_iters") or 2

	review_prompt = """
	You are reviewing the plan for a research agent for GENERALITY and COMPATIBILITY.

	Goals:
	- Be minimal and general; avoid restrictive tool signatures or fixed schemas
	- Prefer simple, user-facing parameters (dataset names, basic scalars)
	- Encourage clear descriptions that surface expectations without enforcing formats
	- Ensure tools can work together, and avoid overlap among colleagues
	- If issues exist, propose revised tool implementation commands that are more general

	Plan to review:
	- Main system prompt: {main_system_prompt}
	- Colleagues: {colleagues}
	- Tool implementations: {tool_implementations}
	"""

	structured_model = model_reviewer.with_structured_output(GraphReviewVerdict, method="json_schema", strict=True)
	response = structured_model.invoke([
		HumanMessage(content=review_prompt.format(
			main_system_prompt=state.get("main_system_prompt", ""),
			colleagues=state.get("colleagues", []),
			tool_implementations=state.get("tool_implementations", []),
		))
	])

	if response.approved or iterations >= max_iters:
		updated_impl = response.revised_tool_implementations or state.get("tool_implementations", [])
		return Command(
			goto="build_compat_schema",
			update={
				"tool_implementations": updated_impl,
				"graph_review_iterations": iterations,
				"messages": [AIMessage(content=("Plan approved." if response.approved else "Reached review limit; proceeding with best plan."))]
			}
		)
	else:
		return Command(
			goto="graph_builder",
			update={
				"graph_review_iterations": iterations + 1,
				"messages": [AIMessage(content=f"Graph review critique: {response.critique}\nRequested changes: {response.required_changes}")]
			}
		)


def build_compat_schema(state: DynamicAgentState) -> Command[Literal["__end__"]]:
	"""Create a general, non-binding compatibility schema and attach to state"""
	project_dir = Path(state.get("project_dir") or "projects/default_researcher")
	project_dir.mkdir(parents=True, exist_ok=True)
	schema_path = project_dir / "compatibility_schema.json"

	schema = {
		"purpose": "General guidelines to encourage flexible, compatible tools",
		"inputs": "Prefer dataset-name strings and simple scalars; avoid passing large objects",
		"db_access": "All data via db.get_table/save_table; no external I/O",
		"returns": "Always return concise human-readable strings; optional diagnostics may be saved",
		"filesystem_outputs": "If saving plots/images/tables, write only under config['output_path'] (projects/<slug>/outputs/). Do not read from this directory; it is write-only for visualization artifacts.",
		"errors": "Provide actionable messages; suggest overrides; avoid enforcing strict schemas",
		"naming": "Encourage clear, deterministic artifact names when saving; versioning suggested",
		"privacy_security": "No secrets in returns/logs; treat config as read-only",
	}
	schema_path.write_text(json.dumps(schema, indent=2))

	return Command(
		goto="__end__",
		update={
			"compatibility_schema": json.dumps(schema),
			"compatibility_schema_path": str(schema_path),
		}
	)
