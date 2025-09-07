from typing_extensions import Literal, List
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain.chat_models import init_chat_model
from langgraph.types import Command, Send
from src.builder_states import DynamicAgentState, GraphPlan
from pydantic import BaseModel, Field

class ClarificationResponse(BaseModel):
    """Schema for user clarification decisions"""
    need_clarification: bool = Field(description="Whether more clarification is needed")
    clarifying_question: str = Field(description="Question to ask user for more details")
    research_requirements: str = Field(description="Complete research requirements if no clarification needed")

model = init_chat_model(model="anthropic:claude-sonnet-4-20250514", max_tokens=4096)

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
    
    structured_model = model.with_structured_output(ClarificationResponse)
    
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

def graph_builder(state: DynamicAgentState) -> Command[Literal["tool_builder", "__end__"]]:
    """Design the research agent architecture"""
    planning_prompt = """
    You are designing a research agent architecture.
    
    Design the MINIMAL but effective research team following these principles:
    1. Start with just a main reasoning orchestrator
    2. Only add colleagues if they significantly improve performance
    3. Each colleague should have distinct expertise
    4. Minimize tool overlap between colleagues
    5. Prefer using predefined tools when possible

    By default, the research agent will have a reasoning orchestrator and a database manager. These should NOT be added to the colleagues list. 
    - The database manager is completely predefined. 
    - The reasoning orchestrator needs a system prompt with a task, a role, and a list of colleagues.
    
    Predefined tools (available to all nodes by default, dont specify): 
    {predefined_tools}
    
    For any tools not in the predefined list, you must create implementation commands. Be extremely thorough in your implementation commands. You must ensure that the tools are complete and can be used intuitively by the research agent.
    Each tool will be implemented in paralell. You must therefore make sure all arguments, data types and return types are specified and compatible.
    The research agent stores and reads data from a database, which is automatically provided to the tools. Any datasets used or written by a tool will be passed as key (str) to the tool. 
    Ensuring compatibility between tools is crucial and should be enforced gracefully using informative error messages, and forcefully by restricting the tools to only accept absolutely nessecary arguments. For example, if a model needs to train on a dataset, but the user has specified that we only need a single dataset throughout the research, then you may predefine the dataset name as "training_dataset" in all tools and not allow any other dataset names.
    Any such predefined schemas also need to be described in the system prompt of the research agent.
    
    EXAMPLES:
    {graph_examples}

    Now, design an agent based on the following requirements:
    {research_requirements}
    """
    
    structured_model = model.with_structured_output(GraphPlan)
    
    response = structured_model.invoke([
        HumanMessage(content=planning_prompt.format(
            research_requirements=state["research_requirements"],
            predefined_tools=json.dumps(PREDEFINED_TOOLS, indent=2),
            graph_examples=get_examples()
        ))
    ])
    
    if response.tool_implementations:
        # Store the state for finalization and use Send to distribute tool building
        updated_state = {
            "colleagues": [colleague.dict() for colleague in response.colleagues],
            "tool_implementations": response.tool_implementations,
            "main_system_prompt": response.main_system_prompt,
            "messages": [AIMessage(content=f"Building {len(response.tool_implementations)} custom tools in parallel...")]
        }
        return Command(
            goto="finalize_agent",
            update=updated_state
        )
    else:
        # No custom tools needed, can proceed directly to final output
        final_spec = {
            "main_system_prompt": response.main_system_prompt,
            "colleagues": [colleague.dict() for colleague in response.colleagues]
        }
        
        return Command(
            goto="__end__",
            update={
                "final_agent_spec": json.dumps(final_spec, indent=2),
                "messages": [AIMessage(content=f"Research agent specification complete!\n\n```json\n{json.dumps(final_spec, indent=2)}\n```")]
            }
        )