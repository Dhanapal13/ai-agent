"""Entry point for running the PR multi-agent system."""
from typing import Optional

from langchain_core.messages import HumanMessage

from state import PRAgentState
from graph import app


def process_pr_with_agent(
    repo_name: str,
    pr_number: int,
    pr_url: str,
    admin_email: Optional[str] = None,
) -> dict:
    """Run the fetch -> risk -> email pipeline for a single PR."""
    initial_state: PRAgentState = {
        "messages": [
            HumanMessage(
                content=(
                    f"New PR received.\nRepository: {repo_name}\n"
                    f"PR Number: {pr_number}\nPR URL: {pr_url}"
                )
            )
        ],
        "repo": repo_name,
        "pr_number": pr_number,
        "pr_url": pr_url,
        "admin_email": admin_email or "",
        "pr_details": None,
        "fetch_error": None,
        "risk_level": None,
        "risk_reasoning": None,
        "email_sent": False,
        "email_error": None,
    }

    config = {"configurable": {"thread_id": f"{repo_name}#{pr_number}"}}

    print(f"Multi-agent PR pipeline started for PR #{pr_number}")

    try:
        result = app.invoke(initial_state, config=config)
        print(f"Pipeline completed for PR #{pr_number}")
        return {
            "status": "success",
            "pr_number": pr_number,
            "repo_name": repo_name,
            "risk_level": result.get("risk_level"),
            "risk_reasoning": result.get("risk_reasoning"),
            "email_sent": result.get("email_sent"),
            "messages": result["messages"],
        }
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed for PR #{pr_number} in {repo_name}: {exc}")
        return {
            "status": "error",
            "pr_number": pr_number,
            "repo_name": repo_name,
            "error": str(exc),
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python main.py <owner/repo> <pr_number> <pr_url>")
        sys.exit(1)

    repo_arg, pr_number_arg, pr_url_arg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    output = process_pr_with_agent(repo_arg, pr_number_arg, pr_url_arg)
    print(output)