# Create Agent class to handle PR processing logic
from datetime import datetime
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent   # ← Fixed import
from langchain_core.prompts import ChatPromptTemplate
from tools import TOOLS

load_dotenv()


def create_pr_agent() -> AgentExecutor:
    """
    Create a LangChain agent for processing GitHub PRs.
    """
    # Initialize the Ollama LLM
    llm = ChatOllama(
        model="llama3.2",                    # Fixed: usually "llama3.2" not "ollama3.2"
        temperature=0.7,
        num_ctx=8192,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # System Prompt
    system_prompt = """You are an expert GitHub Pull Requests (PRs) Security Analyst and reviewer agent.
Your job is to analyze PRs intelligently and take appropriate action.

You have access to the following tools:
1. Fetch PR Details
2. Assess PR Risk
3. Send mail

Rules:
- Always fetch PR details first before making any assessment.
- Use the Assess PR Risk tool to evaluate security implications.
- Be professional, constructive, and objective in your communication.
- If something fails, explain the failure clearly and suggest next steps."""

    # Human Prompt Template
    human_prompt = """A new PR has been created or updated. Analyze it and take appropriate actions.

Repository: {repo}
PR Number: {pr_number}
GitHub URL: {pr_url}
Current Date: {current_date}

Please follow these steps:
1. Fetch the PR details using the Fetch PR Details tool.
2. Assess the security risk using the Assess PR Risk tool.
3. Based on your assessment, send an email using the Send mail tool with your findings and recommendations."""

    # Correct Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", human_prompt),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Create the agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt
    )

    # Create AgentExecutor
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=True
    )


def process_pr_with_agent(agent: AgentExecutor, repo_name: str, pr_number: int, pr_url: str) -> dict:
    """
    Process a GitHub PR using the provided agent.
    """
    task_input = {
        "repo": repo_name,
        "pr_number": pr_number,
        "pr_url": pr_url,
        "current_date": datetime.now().isoformat()
    }

    print(f"🤖 Tool-Calling Agent starting for PR #{pr_number} in {repo_name}")

    try:
        # Use .invoke() instead of .run() (newer LangChain)
        result = agent.invoke(task_input)

        return {
            "status": "success",
            "pr_number": pr_number,
            "repo_name": repo_name,
            "result": result,
            "agent_output": result.get("output", ""),
            "intermediate_steps": result.get("intermediate_steps", [])
        }

    except Exception as e:
        print(f"❌ Agent execution failed for PR #{pr_number} in {repo_name}: {str(e)}")
        return {
            "status": "error",
            "pr_number": pr_number,
            "repo_name": repo_name,
            "error": str(e)
        }