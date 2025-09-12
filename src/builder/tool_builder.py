"""
Standalone Tool Builder Agent

This module implements an independent tool builder agent that can be compiled
separately and used with LangGraph's Send API. Each instance builds a single
tool following strict database schema requirements.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from src.builder.builder_states import ToolBuildState

# Unified review verdict for both single and batch
class ReviewVerdict(BaseModel):
	approved: bool
	score: int = Field(ge=0, le=10, description="0-10 quality score")
	critique: str = ""
	required_changes: str = ""

	class Config:
		extra = "forbid"

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ToolImplementation(BaseModel):
	"""Schema for tool implementation output with examples"""
	tool_name: str = Field(description="Name of the implemented tool")
	tool_code: str = Field(description="Complete Python code for the tool")
	tool_description: str = Field(description="Brief description of what the tool does")
	example_1: str = Field(description="First example of how to use the tool")
	example_2: str = Field(description="Second example of how to use the tool")
	data_inputs: str = Field(description="Description of expected data inputs as dataset names")
	data_outputs: str = Field(description="Description of what data is saved to database")

class ToolImplementationLoose(BaseModel):
	"""Relaxed schema for providers that may omit fields"""
	tool_name: Optional[str] = None
	tool_code: Optional[str] = None
	tool_description: Optional[str] = None
	example_1: Optional[str] = None
	example_2: Optional[str] = None
	data_inputs: Optional[str] = None
	data_outputs: Optional[str] = None

	class Config:
		extra = "allow"

# ===== CONFIGURATION =====

MODEL_ID = "anthropic:claude-sonnet-4-20250514"
model = init_chat_model(MODEL_ID, temperature=0, max_tokens=8096)

# ===== TOOL BUILDING NODES =====

def _fallback_generate_tool_json(implementation_command: str, tool_name: str) -> ToolImplementation:
	"""Fallback: ask for raw JSON matching the ToolImplementation schema and parse it."""
	schema = ToolImplementation.model_json_schema()
	prompt = (
		"You failed to provide all required fields in the previous response.\n"
		"Return ONLY a JSON object that strictly matches this JSON Schema.\n"
		"Fill EVERY required field. Do not include extra keys.\n\n"
		f"JSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
		"Instructions:\n"
		f"- tool_name must be '{tool_name}'.\n"
		"- tool_code must be a complete Python function implementation string that follows the required signature and behavior.\n"
		"- tool_description succinctly describes the tool.\n"
		"- example_1 and example_2 show realistic usage (as strings).\n"
		"- data_inputs and data_outputs describe dataset-name inputs and saved artifacts.\n\n"
		f"Context:\nTool Name: {tool_name}\nImplementation Command: {implementation_command}\n"
	)
	msg = HumanMessage(content=prompt)
	raw = model.invoke([msg])
	# Some providers wrap JSON in code fences – extract braces region if needed
	text = getattr(raw, "content", raw)
	start = text.find("{")
	end = text.rfind("}")
	json_str = text[start:end+1] if start != -1 and end != -1 else text
	data = json.loads(json_str)
	return ToolImplementation(**data)


def _generate_batch_tools_json(commands: List[str], names: List[str], prior_map: Dict[str, str]) -> List[Dict[str, Any]]:
	"""Single-pass batch tool generation: returns list[dict] for all tools in one JSON array."""
	batch_lines = [
		"Implement ALL tools below in a unified, compatible way.",
		"Return ONLY a JSON array where each item has keys: tool_name, tool_code, tool_description, example_1, example_2, data_inputs, data_outputs.",
		"Do not include extra keys or any text outside the JSON array.",
		"\nTOOLS:",
	]
	for cmd, nm in zip(commands, names):
		feedback = prior_map.get(nm, "")
		line = f"- {nm}: {cmd}"
		if feedback:
			line += f" | Incorporate: {feedback}"
		batch_lines.append(line)
	msg = HumanMessage(content="\n".join(batch_lines))
	raw = model.invoke([msg])
	text = getattr(raw, "content", raw)
	start = text.find("[")
	end = text.rfind("]")
	json_str = text[start:end+1] if start != -1 and end != -1 else text
	try:
		arr = json.loads(json_str)
	except Exception:
		arr = []
	# Normalize
	result: List[Dict[str, Any]] = []
	for item in arr:
		if isinstance(item, dict):
			item.setdefault("tool_name", "")
			item.setdefault("tool_code", "")
			item.setdefault("tool_description", "")
			item.setdefault("example_1", "")
			item.setdefault("example_2", "")
			item.setdefault("data_inputs", "")
			item.setdefault("data_outputs", "")
			result.append(item)
	return result

def implement_tool(state: ToolBuildState):
	"""
	Implement a research tool based on the specification.

	Creates a tool following strict requirements:
	- Database schema compliance
	- String-only returns
	- Proper error handling
	- Tool card with examples
	"""

	implementation_prompt = """
	You are implementing a research tool with this specification:

	Tool Name: {tool_name}
	Implementation Command: {implementation_command}

	Only implement functional tools that perform actions on datasets or save artifacts.

	CRITICAL REQUIREMENTS for all tools:

	1. FUNCTION SIGNATURE must be: def tool_name(<user_args...>, *, db, config) -> str
	   - Declare user-facing arguments explicitly (e.g., dataset_name: str, group_col: str)
	   - Use these parameters directly; do not assume undeclared variables

	2. DATABASE INTERACTION:
	   - All data access through db parameter (PyArrow database)
	   - All system access through config parameter (dict) such as gpu availability, memory, etc if applicable
	   - Do not include db and config in function description
	   - Load datasets by name: dataset = db.get_table(dataset_name)
	   - Save results with clear naming: db.save_table(f"result_{{dataset_name}}", result_data)
	   - Only dataset NAMES (strings) are passed as arguments, not actual data
	   - If saving plots/images/tables, write ONLY under config['output_path']; do not read from this directory

	DESIGN PRINCIPLES (brief):
	- DB-oriented only: read/write via db; no external fetching in analysis tools.
	- Do NOT do data generation/formatting inside tools. Use the Data Formatter node for prep; split prep and analysis/modeling into separate tools.

	3. RETURN REQUIREMENTS:
	   - Always return a descriptive string about what was accomplished
	   - Never return actual data or large objects unless explicitly specified
	   - Include key metrics, findings, or confirmations in the return string

	4. ERROR HANDLING:
	   - Handle missing datasets gracefully
	   - Validate input parameters
	   - Be type-robust: attempt sensible conversions (e.g., strings→numeric; strings→categorical encodings) and handle unexpected types gracefully
	   - Return informative messages describing expected types/shapes and what was observed when conversions fail

	5. TOOL CARD:
	   - Provide 2 realistic usage examples
	   - Describe expected inputs and outputs clearly
	   - Include performance considerations if relevant
	   - In both the description and the function docstring, explicitly note expected shapes/types for datasets and columns (e.g., numeric vs categorical, expected data types, typical dimensions) without enforcing hard schemas.

	Keep the implementation general and flexible. Avoid overly restrictive schemas; surface expectations in the description.
	"""

	# Try relaxed structured output first (better for Sonnet-4), then coerce to strict; fallback to JSON if empty
	compat_notes = state.get("compatibility_schema", "")
	cmds = state.get("implementation_commands") or []
	names = state.get("tool_names") or []
	impl_cmd = (cmds[0] if cmds else (state.get("implementation_command") or "")).strip()
	tool_name = (names[0] if names else (state.get("tool_name") or "tool")).strip()
	content = implementation_prompt.format(
		implementation_command=impl_cmd,
		tool_name=tool_name
	)
	if compat_notes:
		content += "\n\nCompatibility guidelines (non-binding):\n" + str(compat_notes)
	loose_model = model.with_structured_output(ToolImplementationLoose)
	loose = loose_model.invoke([HumanMessage(content=content)])
	strict = ToolImplementation(
		tool_name=loose.tool_name or tool_name,
		tool_code=loose.tool_code or "",
		tool_description=loose.tool_description or "",
		example_1=loose.example_1 or "",
		example_2=loose.example_2 or "",
		data_inputs=loose.data_inputs or "",
		data_outputs=loose.data_outputs or "",
	)
	if strict.tool_code.strip() == "" or strict.tool_description.strip() == "":
		strict = _fallback_generate_tool_json(impl_cmd, tool_name)

	tool_dict = {
		"tool_name": strict.tool_name,
		"tool_code": strict.tool_code,
		"tool_description": strict.tool_description,
		"tool_card": {
			"name": strict.tool_name,
			"description": strict.tool_description,
			"examples": [strict.example_1, strict.example_2],
			"data_inputs": strict.data_inputs,
			"data_outputs": strict.data_outputs,
		},
	}
	return {"tools": [tool_dict]}


def implement_tool_batch(state: ToolBuildState):
	"""
	Implement a batch of tools (up to 5) in ONE pass to unify schema and ensure compatibility.
	"""
	commands = state.get("implementation_commands") or []
	names = state.get("tool_names") or []
	max_len = min(len(commands), len(names), 5)

	prior_map: Dict[str, str] = {}
	if state.get("attempts"):
		for nm in names[:max_len]:
			attempts = state["attempts"].get(nm, [])
			if attempts:
				prior_map[nm] = attempts[-1].get("required_changes", "") or attempts[-1].get("critique", "")

	items = _generate_batch_tools_json(commands[:max_len], names[:max_len], prior_map)
	tools: List[Dict[str, Any]] = []
	for item in items:
		name = item.get("tool_name", "")
		tools.append({
			"tool_name": name,
			"tool_code": item.get("tool_code", ""),
			"tool_description": item.get("tool_description", ""),
			"tool_card": {
				"name": name,
				"description": item.get("tool_description", ""),
				"examples": [item.get("example_1", ""), item.get("example_2", "")],
				"data_inputs": item.get("data_inputs", ""),
				"data_outputs": item.get("data_outputs", ""),
			},
		})
	return {"tools": tools}


def review_tool_batch(state: ToolBuildState):
	"""Batch review with a single overall verdict for compatibility and faithfulness."""
	iterations = state.get("review_iterations") or 0
	max_iters = 3
	tools_list = state.get("tools") or []
	tools_payload = [{"name": t["tool_name"], "description": t["tool_description"], "code": t["tool_code"]} for t in tools_list]
	impl_cmds = state.get("implementation_commands", [])
	compat_notes = state.get("compatibility_schema", "")
	prompt = """
	You are the Tool Reviewer. Evaluate the ENTIRE SET of tools together for overall correctness, clarity, and cross-tool compatibility.

	REQUIREMENTS (must verify across the set):
	- All tools follow: def tool_name(<user_args...>, *, db, config) -> str
	- User-facing args are explicit; no undeclared variables
	- I/O only via db.get_table/save_table; inputs are dataset-name strings or simple scalars
	- Returns are concise human-readable strings; helpful, non-restrictive errors
	- If writing plots/images/tables, they save ONLY under config['output_path'] and never read from it
	- Descriptions and docstrings explicitly state expected data/column shapes and types (e.g., required columns, numeric/categorical), phrased as expectations not rigid schemas
	- Feature transforms/encodings inside analysis tools (e.g., scaling, one-hot) are allowed; avoid general data generation/formatting which belongs in the Data Formatter
	- Tools are type-compatible: they attempt reasonable type conversions and report clear guidance when types don't match expectations
	- Prefer separate dataset names for training, prediction/evaluation, and outputs when applicable

	Builder requests (implementation_commands): {impl_cmds}
	Compatibility guidelines (non-binding): {compat_notes}

	Deliverables (OVERALL verdict for the set): approved (bool), score (0-10), critique, required_changes.

	Tools to review (name/desc/code payload): {tools}
	"""
	# Create a fresh non-streaming client for structured output to avoid ResponseNotRead in concurrent calls
	local_model = init_chat_model(MODEL_ID, temperature=0, max_tokens=8096)
	structured_model = local_model.with_structured_output(ReviewVerdict, method="json_schema", strict=True)
	v = structured_model.invoke([
		HumanMessage(content=prompt.format(tools=tools_payload, impl_cmds=impl_cmds, compat_notes=compat_notes))
	])
	new_iterations = (iterations or 0) + 1
	return {
		"review_iterations": new_iterations,
		"approved": v.approved,
		"limit_reached": new_iterations >= max_iters,
	}


def review_tool_single(state: ToolBuildState):
	"""Single-tool review: evaluate one tool and set approved flag."""
	iterations = state.get("review_iterations") or 0
	max_iters = 3
	tools_list = state.get("tools") or []
	tool = tools_list[0]
	t = {"name": tool["tool_name"], "description": tool.get("tool_description"), "code": tool.get("tool_code")}
	impl_cmd = (state.get("implementation_commands") or [""])[0]
	compat_notes = state.get("compatibility_schema", "")
	prompt = """
	You are the Tool Reviewer. Evaluate this tool for correctness, clarity, and compatibility.

	REQUIREMENTS (must verify):
	- Signature: def tool_name(<user_args...>, *, db, config) -> str
	- User-facing args declared explicitly; no undeclared variables
	- I/O via db only (db.get_table / db.save_table); inputs are dataset-name strings or simple scalars
	- Returns a concise human-readable string; helpful, non-restrictive errors
	- If writing plots/images/tables, save ONLY under config['output_path']; do not read from this directory
	- Description and function docstring include expected data/column shapes and types (e.g., required columns, numeric/categorical), expressed as guidance not strict validation
	- Tool is DB-oriented and respects separation of concerns (data generation/management vs analysis/modeling)
	- Tool is type-compatible: it attempts sensible conversions (e.g., strings→numeric) and reports clear guidance when conversions fail
	- Prefer separate dataset names for training, prediction/evaluation, and outputs when applicable

	Builder request (implementation_command): {impl_cmd}
	Compatibility guidelines (non-binding): {compat_notes}

	Deliverables: approved (bool), score (0-10), critique, required_changes. If approved is true, you may leave critique and required_changes empty.

	Tool to review (name/desc/code payload): {tool}
	"""
	# Create a fresh non-streaming client for structured output to avoid ResponseNotRead in concurrent calls
	local_model = init_chat_model(MODEL_ID, temperature=0, max_tokens=8096)
	structured_model = local_model.with_structured_output(ReviewVerdict, method="json_schema", strict=True)
	v = structured_model.invoke([
		HumanMessage(content=prompt.format(tool=t, impl_cmd=impl_cmd, compat_notes=compat_notes))
	])
	attempts = state.get("attempts") or {}
	tool_payload = tools_list[0]
	tool_key = tool_payload.get("tool_name") or t.get("name") or "unknown_tool"
	attempts.setdefault(tool_key, []).append({
		"score": v.score,
		"approved": v.approved,
		"tool": tool_payload,
		"critique": v.critique,
		"required_changes": v.required_changes,
	})
	new_iterations = (iterations or 0) + 1
	return {
		"attempts": attempts,
		"review_iterations": new_iterations,
		"approved": v.approved,
		"limit_reached": new_iterations >= max_iters,
	}


def register_tool(state: ToolBuildState):
	"""
	Register the completed tool to the tool registry under the project directory.
	"""

	try:
		project_dir = Path(state.get("project_dir"))
		tool_registry_path = project_dir / "generated_tools"
		tool_registry_path.mkdir(parents=True, exist_ok=True)

		# Decide final tool(s) to register using best scored attempts if present
		tools_to_register: List[Dict[str, Any]] = []
		if state.get("attempts"):
			for name, tries in state["attempts"].items():
				best = max(tries, key=lambda x: x.get("score", 0))
				tools_to_register.append(best["tool"])
		elif state.get("tools"):
			tools_to_register.extend(state["tools"])  # No scoring info

		completed_tools: List[Dict[str, Any]] = []
		for t in tools_to_register:
			tool_name = t["tool_name"]
			tool_code = t["tool_code"]
			tool_card = t["tool_card"]

			# Basic sanity check: ensure tool code defines a function with the tool's name
			defines_func = isinstance(tool_code, str) and (f"def {tool_name}(" in tool_code)
			if not defines_func:
				continue

			tool_file = tool_registry_path / f"{tool_name}.py"
			full_tool_code = f'''"""
Tool: {tool_name}
Generated by Tool Builder Agent

{tool_card["description"]}

Examples:
1. {tool_card["examples"][0]}
2. {tool_card["examples"][1]}

Data Inputs: {tool_card["data_inputs"]}
Data Outputs: {tool_card["data_outputs"]}
"""

import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Any

# Tool implementation
{tool_code}

# Tool metadata for registration
TOOL_METADATA = {tool_card}
'''
			tool_file.write_text(full_tool_code)

			card_file = tool_registry_path / f"{tool_name}_card.json"
			card_file.write_text(json.dumps(tool_card, indent=2))

			completed_tools.append({
				"name": tool_name,
				"description": tool_card["description"],
				"file_path": str(tool_file),
				"card_path": str(card_file),
				"examples": tool_card["examples"],
			})
		return {"completed_tool": completed_tools}

	except Exception as e:
		return {
			"completed_tool": {
				"name": tool_name,
				"description": f"Error registering tool: {str(e)}",
				"file_path": None,
				"card_path": None,
				"examples": [],
			}
		}


# ===== GRAPH CONSTRUCTION =====

def _next_after_review(state: ToolBuildState) -> str:
	"""Route to implement again or register based on approval/limits"""
	# Respect explicit limit or iteration cap
	if (state.get("review_iterations") or 0) >= 3:
		return "register_tool"
	if state.get("limit_reached"):
		return "register_tool"
	# Only continue implementing when we are explicitly NOT approved
	approved = state.get("approved")
	if approved is False:
		count = len(state.get("tools") or [])
		return "implement_tool_batch" if count > 1 else "implement_tool"
	# Default to registering when approval is True or missing/ambiguous
	return "register_tool"


def _route_mode(state: ToolBuildState) -> str:
	"""Route to batch or single implementation node based on batch_mode flag"""
	cmd_count = len(state.get("implementation_commands") or [])
	name_count = len(state.get("tool_names") or [])
	return "implement_tool_batch" if max(cmd_count, name_count) > 1 else "implement_tool"


def create_tool_builder_agent():
	"""Create the standalone tool builder agent"""

	builder = StateGraph(ToolBuildState)
	builder.add_node("implement_tool", implement_tool)
	builder.add_node("implement_tool_batch", implement_tool_batch)
	builder.add_node("review_tool_single", review_tool_single)
	builder.add_node("review_tool_batch", review_tool_batch)
	builder.add_node("register_tool", register_tool)
	builder.add_conditional_edges(START, _route_mode, ["implement_tool", "implement_tool_batch"])
	builder.add_edge("implement_tool", "review_tool_single")
	builder.add_edge("implement_tool_batch", "review_tool_batch")
	builder.add_conditional_edges("review_tool_single", _next_after_review, ["implement_tool", "register_tool"])
	builder.add_conditional_edges("review_tool_batch", _next_after_review, ["implement_tool_batch", "register_tool"])
	builder.add_edge("register_tool", END)
	return builder.compile()
