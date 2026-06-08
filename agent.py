# Langgraph powered PR Agent

import os
from datetime import datetime
from typing import Annotated, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from tools import TOOLS

load_dotenv()

# Agent State

class PRAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    repo: str
    pr_number: int
    pr_url: str
    pr_details: str
    risk_level: str
    email_send: bool

# System Prompt

SYSTEM_PROMPT = """ You are an expert Github PR reviewer.

Your job is to analysis the PR and the following steps in OREDER:

1. Call 'fetch_pr_details' with repo name and PR number to get PR details
2. Call 'assess_pr_risk' with PR details JSON to determine the risk level (LOW - MEDIUM - HIGH)
3. Call 'send_email' to to notify the admin about PR risk
    - A short summary of PR (Title, author, branch... etc)
    - The risk level and reasoning behind it

Rules:
1. Always fect PR details before making any assessment.
2. Be professional.
3. If tool call fails, provide the details
4. Don't skip any step.

"""

# LLM Init

def create_llm() -> None:
    llm = ChatOllama(
        model= os.getenv("OLLAMA_MODEL", "llama3.2"),
        temperature=0.7,
        num_ctx=8192,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    return llm.bind_tools(TOOLS)

# call llm

def call_llm(agent_state: PRAgentState) -> dict:
    llm = create_llm()
    full_messages = [SystemMessage(content=SYSTEM_PROMPT) + agent_state.get("messages")]
    response: AIMessage = llm.invoke(full_messages)
    return {
        "messages": [response]
    }

def should_continue(state: PRAgentState) -> str:
    last_message = state.get("messages")[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Graph construction

def build_graph(use_checkpoint: bool = False):
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(PRAgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges("llm", should_continue, {
        "tools": "tools",
        END: END
    })

    graph.add_edge("tools", "llm")

    check_pointer = InMemorySaver() if use_checkpoint else None
    return graph.compile(checkpointer=check_pointer)

# Expose as graph as API

app = build_graph(use_checkpoint=True)

def process_pr_with_agent(
        repo_name: str,
        pr_number: int,
        pr_url: str
) -> dict:
    initial_state: PRAgentState = PRAgentState()
    initial_state.messages = [
            HumanMessage(content=
                         f"A new PR has been received. Analysis it and take required action \n. "
                         f"Repository: {repo_name} \n"
                         f"PR Number: {pr_number} \n"
                         f"PR URL: {pr_url} \n"
                         "Please follow these steps." \
                         "1. Fetch PR details by calling fecth_pr_details tool.\n"
                         "2. Assess PR for security risk by calling assess_pr_risk tools. \n"
                         "3. Send mail about PR risk assess using send_mail tool. \n"
                         )
        ]
    initial_state.repo = repo_name
    initial_state.pr_number = pr_number
    initial_state.pr_url = pr_url
    initial_state.pr_details = ""
    initial_state.risk_level =""
    initial_state.email_send = False

    config = {"configurable": {"thread_id": f"{repo_name}#{pr_number}"}}

    print(f"Langgraph agent started for PR - {pr_number}")

    try:
        result = app.invoke(initial_state, config=config)
        final_output = result["messages"][-1].content if result["messages"] else ""
        print(f"Agent completed - {pr_number}")

        return {
            "status": "success",
            "pr_number": pr_number,
            "repo_name": repo_name,
            "agent_output": final_output,
            "messages": result["messages"]

        }

    except Exception as e:
        print(f"❌ Agent failed — PR #{pr_number} in {repo_name}: {e}")
        return {
                    "status": "error",
                    "pr_number": pr_number,
                    "repo_name": repo_name,
                    "error": str(e)
                }
    
def stream_pr_with_agent(repo_name: str, pr_number: int, pr_url: str):
    pass

def visualise_graph():
    pass