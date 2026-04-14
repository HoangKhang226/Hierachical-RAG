"""LangGraph workflow assembly for the Hierarchical RAG pipeline.

This file is the single source of truth for how nodes are connected.
All node logic lives in node.py; all routing/tool logic lives in tool.py.

Flow
----
START
  │
  ▼
context_compressor ──► ambiguity_checker
                              │
                   ┌──────────┴──────────┐
             ambiguous                 clear
                │                       │
               END               planner (sub_tasks)
                                        │
                         Send — fan-out, one branch per sub-task (parallel)
                                        │
                             [subtask_runner × N]
                              knowledge_router
                              ├── "rag" ──► hyde ──► rag_retriever ──► validator
                              │                                   ├── valid   → END
                              │                                   └── invalid → web_searcher
                              ├── "web" ──────────────────────────► web_searcher
                              └── "llm_knowledge" ────────────────► llm_node
                                        │
                              Fan-in — all branches converge
                                        │
                                  global_summary ──► synthesizer ──► END
"""

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState

# Node functions
from src.agents.node import (
    context_compressor,
    ambiguity_checker,
    planner,
    knowledge_router,
    hyde,
    validator,
    global_summary,
    synthesizer,
)

# Routing helpers and retrieval tools
from src.agents.tool import (
    route_after_ambiguity,
    route_after_router,
    route_after_validator,
    fan_out_subtasks,
    rag_retriever,
    web_searcher,
    llm_node,
)


# ---------------------------------------------------------------------------
# Sub-task subgraph  (compiled once, reused per Send invocation)
# ---------------------------------------------------------------------------


def build_subtask_subgraph():
    """Compile the per-sub-task retrieval mini-pipeline.

    Each parallel branch spawned by fan_out_subtasks runs this subgraph
    independently against its own ``current_task``.
    """
    sg = StateGraph(AgentState)

    sg.add_node("knowledge_router", knowledge_router)
    sg.add_node("hyde", hyde)
    sg.add_node("rag_retriever", rag_retriever)
    sg.add_node("validator", validator)
    sg.add_node("web_searcher", web_searcher)
    sg.add_node("llm_node", llm_node)

    sg.set_entry_point("knowledge_router")

    sg.add_conditional_edges(
        "knowledge_router",
        route_after_router,
        {"hyde": "hyde", "web_searcher": "web_searcher", "llm_node": "llm_node"},
    )

    sg.add_edge("hyde", "rag_retriever")
    sg.add_edge("rag_retriever", "validator")
    sg.add_conditional_edges(
        "validator",
        route_after_validator,
        {"web_searcher": "web_searcher", END: END},
    )

    sg.add_edge("web_searcher", END)
    sg.add_edge("llm_node", END)

    return sg.compile()


# ---------------------------------------------------------------------------
# Main graph
# ---------------------------------------------------------------------------


def build_graph():
    """Assemble and compile the top-level Hierarchical RAG graph.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    subtask_subgraph = build_subtask_subgraph()

    g = StateGraph(AgentState)

    g.add_node("context_compressor", context_compressor)
    g.add_node("ambiguity_checker", ambiguity_checker)
    g.add_node("planner", planner)
    g.add_node("subtask_runner", subtask_subgraph)
    g.add_node("global_summary", global_summary)
    g.add_node("synthesizer", synthesizer)

    g.set_entry_point("context_compressor")
    g.add_edge("context_compressor", "ambiguity_checker")

    g.add_conditional_edges(
        "ambiguity_checker",
        route_after_ambiguity,
        {"planner": "planner", END: END},
    )

    # Fan-out: each sub-task runs subtask_runner in parallel
    g.add_conditional_edges("planner", fan_out_subtasks, ["subtask_runner"])

    # Fan-in: all branches meet at global_summary
    g.add_edge("subtask_runner", "global_summary")
    g.add_edge("global_summary", "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()


# Module-level compiled graph — import this in the API layer
graph = build_graph()
