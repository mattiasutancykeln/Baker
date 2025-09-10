"""
Standalone Tool Builder Agent

This module implements an independent tool builder agent that can be compiled
separately and used with LangGraph's Send API. Each instance builds a single
tool following strict database schema requirements.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from src.builder.builder_states import ToolBuildState

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

# ===== CONFIGURATION =====

model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", max_tokens=4096)

# ===== TOOL BUILDING NODES =====

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

    CRITICAL REQUIREMENTS for all tools:

    1. FUNCTION SIGNATURE must be: def tool_name(*, db, config) -> str

    2. DATABASE INTERACTION:
       - All data access through db parameter (PyArrow database)
       - All system access through config parameter (dict) such as gpu availability, memory, etc if applicable
       - Do not include db and config in function description
       - Load datasets by name: dataset = db.get_table(dataset_name)
       - Save results with clear naming: db.save_table(f"result_{{dataset_name}}", result_data)
       - Only dataset NAMES (strings) are passed as arguments, not actual data

    3. RETURN REQUIREMENTS:
       - Always return a descriptive string about what was accomplished
       - Never return actual data or large objects unless explicitly specified
       - Include key metrics, findings, or confirmations in the return string

    4. ERROR HANDLING:
       - Handle missing datasets gracefully
       - Validate input parameters
       - Return informative error messages as strings

    5. TOOL CARD:
       - Provide 2 realistic usage examples
       - Describe expected inputs and outputs clearly
       - Include performance considerations if relevant

    Example implementation pattern:
    ```python
    def correlation_matrix(dataset_name, db, config) -> str:
        '''Compute correlation matrix for numeric columns in dataset.
        Args:
            dataset_name: Name of dataset in database (str)
        Returns:
            String description of correlation findings
        Examples:
            result = correlation_matrix(dataset_name="customer_data")
            result = correlation_matrix(dataset_name="survey_responses")
        '''
        try:
            # Load dataset from database
            if not db.has_table(dataset_name):
                return f"Error: Dataset '{{dataset_name}}' not found in database"
            dataset = db.get_table(dataset_name).to_pandas()
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return f"Error: Need at least 2 numeric columns, found {{len(numeric_cols)}}"
            corr_matrix = dataset[numeric_cols].corr()
            db.save_table(f"correlations_{{dataset_name}}", corr_matrix)
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append(f"{{corr_matrix.columns[i]}} - {{corr_matrix.columns[j]}}: {{corr_val:.3f}}")
            return (
                f"Correlation matrix computed for {{len(numeric_cols)}} numeric columns in '{{dataset_name}}'. "
                f"Strong correlations (>0.7): {{('; '.join(strong_corrs) if strong_corrs else 'None found')}}. "
                f"Results saved as 'correlations_{{dataset_name}}'"
            )
        except Exception as e:
            return f"Error computing correlation matrix for '{{dataset_name}}': {{str(e)}}"
    ```

    Implement the requested tool following this exact pattern with proper tool card documentation.
    """

    structured_model = model.with_structured_output(ToolImplementation)

    response = structured_model.invoke([
        HumanMessage(content=implementation_prompt.format(
            implementation_command=state["implementation_command"],
            tool_name=state["tool_name"]
        ))
    ])

    return {
        "tool_name": response.tool_name,
        "tool_code": response.tool_code,
        "tool_description": response.tool_description,
        "tool_card": {
            "name": response.tool_name,
            "description": response.tool_description,
            "examples": [response.example_1, response.example_2],
            "data_inputs": response.data_inputs,
            "data_outputs": response.data_outputs,
        },
    }


def register_tool(state: ToolBuildState):
    """
    Register the completed tool to the tool registry under the project directory.
    """

    tool_name = state["tool_name"]
    tool_code = state["tool_code"]
    tool_card = state["tool_card"]

    try:
        project_dir = Path(state.get("project_dir") or "projects/default_researcher")
        tool_registry_path = project_dir / "generated_tools"
        tool_registry_path.mkdir(parents=True, exist_ok=True)

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

        completed_tool = {
            "name": tool_name,
            "description": tool_card["description"],
            "file_path": str(tool_file),
            "card_path": str(card_file),
            "examples": tool_card["examples"],
        }
        return {"completed_tool": completed_tool}

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

def create_tool_builder_agent():
    """Create the standalone tool builder agent"""

    builder = StateGraph(ToolBuildState)
    builder.add_node("implement_tool", implement_tool)
    builder.add_node("register_tool", register_tool)
    builder.add_edge(START, "implement_tool")
    builder.add_edge("implement_tool", "register_tool")
    builder.add_edge("register_tool", END)
    return builder.compile()
