"""
Standalone Tool Builder Agent

This module implements an independent tool builder agent that can be compiled
separately and used with LangGraph's Send API. Each instance builds a single
tool following strict database schema requirements.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from  src.builder_states import ToolBuildState
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
tool_registry_path = Path("generated_tools")
tool_registry_path.mkdir(exist_ok=True)

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
    
    1. FUNCTION SIGNATURE must be: def tool_name(*, db=database, config=config) -> str
    
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
            
            # Compute correlations
            corr_matrix = dataset[numeric_cols].corr()
            
            # Save to database with clear naming
            db.save_table(f"correlations_{{dataset_name}}", corr_matrix)
            
            # Find strong correlations for summary
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i,j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append(f"{{corr_matrix.columns[i]}} - {{corr_matrix.columns[j]}}: {{corr_val:.3f}}")
            
            return f"Correlation matrix computed for {{len(numeric_cols)}} numeric columns in '{{dataset_name}}'. Strong correlations (>0.7): {{'; '.join(strong_corrs) if strong_corrs else 'None found'}}. Results saved as 'correlations_{{dataset_name}}'"
            
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
            "data_outputs": response.data_outputs
        }
    }

def register_tool(state: ToolBuildState):
    """
    Register the completed tool to the tool registry.
    
    Saves the tool code to file and creates tool card documentation.
    """
    
    tool_name = state["tool_name"]
    tool_code = state["tool_code"]
    tool_card = state["tool_card"]
    
    try:
        # Save tool code to registry
        tool_file = tool_registry_path / f"{tool_name}.py"
        
        # Add necessary imports and database setup to the tool code
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
        
        # Create tool card file
        card_file = tool_registry_path / f"{tool_name}_card.json"
        card_file.write_text(json.dumps(tool_card, indent=2))
        
        # Prepare completed tool specification
        completed_tool = {
            "name": tool_name,
            "description": tool_card["description"], 
            "file_path": str(tool_file),
            "card_path": str(card_file),
            "examples": tool_card["examples"],
        }
        
        return {
            "completed_tool": completed_tool
        }
        
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
    
    # Add nodes
    builder.add_node("implement_tool", implement_tool)
    builder.add_node("register_tool", register_tool)
    
    # Add edges
    builder.add_edge(START, "implement_tool")
    builder.add_edge("implement_tool", "register_tool")
    builder.add_edge("register_tool", END)
    
    return builder.compile()

# ===== USAGE FUNCTIONS =====

def build_single_tool(implementation_command: str, tool_name: str):
    """
    Build a single tool using the tool builder agent.
    
    Args:
        implementation_command: Specification for the tool to build
        tool_name: Name of the tool to build
        
    Returns:
        Dict containing completed tool specification
    """
    
    agent = create_tool_builder_agent()
    
    result = agent.invoke({
        "implementation_command": implementation_command,
        "tool_name": tool_name
    })
    
    return result["completed_tool"]

def build_multiple_tools(implementation_commands: list, tool_names: list):
    """
    Build multiple tools sequentially.
    
    Args:
        implementation_commands: List of tool specifications
        tool_names: List of tool names
        
    Returns:
        List of completed tool specifications
    """
    
    if tool_names is None:
        tool_names = [f"Tool {i+1}" for i in range(len(implementation_commands))]
    
    completed_tools = []
    
    for i, cmd in enumerate(implementation_commands):
        role = tool_names[i] if i < len(tool_names) else f"Tool {i+1}"
        tool = build_single_tool(cmd, role)
        completed_tools.append(tool)
    
    return completed_tools

# ===== EXAMPLE TOOL IMPLEMENTATIONS =====

EXAMPLE_TOOL_COMMANDS = [
    "Implement correlation_significance(dataset, col1, col2) that tests the statistical significance of correlation between two variables, including confidence intervals and sample size considerations.",
    "Implement gp_train_predict(x_train, y_train, x_test, predictions_dataset) that trains a Gaussian Process regressor on specified features and target. Should handle feature scaling, kernel selection, and save the trained model with performance metrics.",
    "Implement ucb_ranking(predictions_dataset, confidence_param=1.96) that ranks entries by Upper Confidence Bound (mean + confidence_param * std) to balance expected value with uncertainty for optimal decision making."
]

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Example: Build all example tools
    print("Building example tools...")
    
    completed_tools = build_multiple_tools(
        EXAMPLE_TOOL_COMMANDS,
        ["correlation_significance", "gp_train_predict", "ucb_ranking"]
    )
    
    print(f"\nSuccessfully built {len(completed_tools)} tools:")
    for tool in completed_tools:
        print(f"- {tool['name']}: {tool['description']}")
        
    print(f"\nTools saved to: {tool_registry_path}")