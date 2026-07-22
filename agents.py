"""The three agents in the PR review multi-agent system.

Each agent is a LangGraph node: it reads what it needs from state, does its
job (calling a deterministic tool, and in most cases asking an LLM to reason
about / write up the result), and returns a partial state update that
LangGraph merges back in. They are chained together in graph.py.

Design choice: rather than making every agent a full ReAct loop (LLM decides
whether to call a tool), each agent deterministically calls the one tool
it's responsible for, then uses the LLM purely for reasoning/summarization.
This keeps the pipeline predictable (Fetch always fetches, Risk always
assesses, Email always attempts to notify) while still using the LLM for the
parts that benefit from it — explaining risk, writing the email.
"""
import os
import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state import PRAgentState
from tools import fetch_pr_details, run_risk_heuristic, send_email

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _llm(temperature: float = 0.3) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=temperature,
        num_ctx=8192,
        base_url=OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Agent 1 — Fetch PR details
# ---------------------------------------------------------------------------

FETCH_AGENT_PROMPT = """You are the PR Intake Agent.
You are given the raw JSON of a GitHub pull request. Write a concise 2-4
sentence summary covering: title, author, source -> target branch, and the
size of the change (files/additions/deletions). Do not invent information
that is not present in the JSON."""


def fetch_agent(state: PRAgentState) -> dict:
    """Agent 1: calls GitHub, then asks the LLM to summarize the PR."""
    repo = state["repo"]
    pr_number = state["pr_number"]

    result = fetch_pr_details(repo, pr_number)

    if "error" in result:
        return {
            "pr_details": None,
            "fetch_error": result["error"],
            "messages": [AIMessage(content=f"[fetch_agent] Failed to fetch PR: {result['error']}")],
        }

    llm = _llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=FETCH_AGENT_PROMPT),
        HumanMessage(content=json.dumps(result, indent=2)),
    ])

    return {
        "pr_details": result,
        "fetch_error": None,
        "messages": [AIMessage(content=f"[fetch_agent] {response.content}")],
    }


# ---------------------------------------------------------------------------
# Agent 2 — Risk analysis
# ---------------------------------------------------------------------------

RISK_AGENT_PROMPT = """You are the PR Risk Analysis Agent.
You are given PR metadata and a keyword-based heuristic signal. Decide a
final risk level of exactly one of: LOW, MEDIUM, HIGH.

Consider:
- The heuristic signal as a starting point, not a verdict.
- The size of the change (more files/lines = more risk).
- Whether changed filenames suggest security-sensitive areas (auth, secrets,
  infra/config, CI, dependencies).

Respond ONLY as JSON with this exact shape, no prose outside the JSON:
{"risk_level": "LOW|MEDIUM|HIGH", "reasoning": "2-3 sentence explanation"}"""


def risk_agent(state: PRAgentState) -> dict:
    """Agent 2: runs the heuristic, then asks the LLM to make the final call."""
    if state.get("fetch_error"):
        return {
            "risk_level": "UNKNOWN",
            "risk_reasoning": "Skipped — PR details were not available.",
            "messages": [AIMessage(content="[risk_agent] Skipped risk analysis: no PR details.")],
        }

    pr_details = state["pr_details"]
    heuristic = run_risk_heuristic(pr_details)

    llm = _llm(temperature=0.2)
    payload = {"pr_details": pr_details, "heuristic_signal": heuristic}
    response = llm.invoke([
        SystemMessage(content=RISK_AGENT_PROMPT),
        HumanMessage(content=json.dumps(payload, indent=2)),
    ])

    risk_level, reasoning = heuristic["heuristic_level"], response.content
    try:
        parsed = json.loads(response.content)
        risk_level = str(parsed.get("risk_level", risk_level)).upper()
        reasoning = parsed.get("reasoning", reasoning)
    except (json.JSONDecodeError, AttributeError):
        # LLM didn't return clean JSON — fall back to the deterministic heuristic
        pass

    return {
        "risk_level": risk_level,
        "risk_reasoning": reasoning,
        "messages": [AIMessage(content=f"[risk_agent] Risk level: {risk_level}. {reasoning}")],
    }


# ---------------------------------------------------------------------------
# Agent 3 — Send email
# ---------------------------------------------------------------------------

EMAIL_AGENT_PROMPT = """You are the Notification Agent.
Write a short, professional email notifying an admin about a reviewed PR.
Include: PR title, author, source -> target branch, link, risk level, and
the risk reasoning. If PR details could not be fetched, say so plainly
instead. Keep it under 150 words.

Respond ONLY as JSON with this exact shape, no prose outside the JSON:
{"subject": "...", "body": "..."}"""


def email_agent(state: PRAgentState) -> dict:
    """Agent 3: drafts and sends the notification email."""
    admin_email = state.get("admin_email") or os.getenv("ADMIN_EMAIL")
    if not admin_email:
        return {
            "email_sent": False,
            "email_error": "No admin_email provided in state or ADMIN_EMAIL env var.",
            "messages": [AIMessage(content="[email_agent] Skipped: no recipient email configured.")],
        }

    context = {
        "repo": state["repo"],
        "pr_number": state["pr_number"],
        "pr_url": state.get("pr_url"),
        "pr_details": state.get("pr_details"),
        "fetch_error": state.get("fetch_error"),
        "risk_level": state.get("risk_level"),
        "risk_reasoning": state.get("risk_reasoning"),
    }

    llm = _llm(temperature=0.4)
    response = llm.invoke([
        SystemMessage(content=EMAIL_AGENT_PROMPT),
        HumanMessage(content=json.dumps(context, indent=2)),
    ])

    subject = f"PR #{state['pr_number']} review — risk: {state.get('risk_level', 'UNKNOWN')}"
    body = response.content
    try:
        parsed = json.loads(response.content)
        subject = parsed.get("subject", subject)
        body = parsed.get("body", body)
    except (json.JSONDecodeError, AttributeError):
        pass  # fall back to the raw LLM text as the body

    result = send_email(admin_email, subject, body)

    if result.get("success"):
        return {
            "email_sent": True,
            "email_error": None,
            "messages": [AIMessage(content=f"[email_agent] Email sent to {admin_email}.")],
        }

    return {
        "email_sent": False,
        "email_error": result.get("error"),
        "messages": [AIMessage(content=f"[email_agent] Failed to send email: {result.get('error')}")],
    }