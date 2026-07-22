"""Shared state definition for the PR multi-agent system."""
from typing import Annotated, Optional, TypedDict
import operator
from langchain_core.messages import BaseMessage


class PRAgentState(TypedDict):
    # Conversation / audit trail contributed to by all three agents
    messages: Annotated[list[BaseMessage], operator.add]

    # Inputs
    repo: str
    pr_number: int
    pr_url: str
    admin_email: str

    # Produced by Agent 1 (Fetch)
    pr_details: Optional[dict]
    fetch_error: Optional[str]

    # Produced by Agent 2 (Risk analysis)
    risk_level: Optional[str]
    risk_reasoning: Optional[str]

    # Produced by Agent 3 (Email)
    email_sent: bool
    email_error: Optional[str]