"""Graph wiring for the PR multi-agent system.

Three specialized agents run in a fixed pipeline:

    fetch_agent -> risk_agent -> email_agent -> END

Each agent is resilient to upstream failures (e.g. risk_agent still runs and
reports "UNKNOWN" if fetch_agent could not reach GitHub, and email_agent
still fires so the admin is told something went wrong), so the pipeline
always completes and always produces a notification.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from state import PRAgentState
from agents import fetch_agent, risk_agent, email_agent


def build_graph(use_checkpoint: bool = False):
    graph = StateGraph(PRAgentState)

    graph.add_node("fetch_agent", fetch_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("email_agent", email_agent)

    graph.set_entry_point("fetch_agent")
    graph.add_edge("fetch_agent", "risk_agent")
    graph.add_edge("risk_agent", "email_agent")
    graph.add_edge("email_agent", END)

    checkpointer = InMemorySaver() if use_checkpoint else None
    return graph.compile(checkpointer=checkpointer)


# Default app instance, exposed for import elsewhere (e.g. an API layer)
app = build_graph(use_checkpoint=True)


def visualise_graph(output_path: str = "pr_agent_graph.png") -> str:
    """Render the graph to a PNG (requires the optional mermaid/graphviz deps)."""
    png_bytes = app.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    return output_path