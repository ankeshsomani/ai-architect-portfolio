import os
from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY from a .env file

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END


model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers"""
    return a / b

tools =[add, subtract, multiply, divide]

tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: dict):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    
    return END

agent_builder=StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)


agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
);

agent_builder.add_edge("tool_node", "llm_call");

agent= agent_builder.compile()

from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

from langchain_core.messages import HumanMessage
messages=[HumanMessage(content = "Add 21 and 3")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

