"""LangGraph state for the Hierarchical RAG agentic workflow.

Node execution order:
  1. ambiguity_checker  → reads: question          | writes: is_ambiguous, rejection_reason
  2. planner            → reads: question          | writes: sub_tasks
  3. knowledge_router   → reads: current_task, content_summary | writes: route
  4. hyde_generator     → reads: current_task      | writes: hyde_query
  5. rag_retriever      → reads: hyde_query        | writes: rag_context
     web_searcher       → reads: current_task      | writes: web_context
  6. context_merger     → reads: rag/web_context   | writes: all_context
  7. validator          → reads: all_context       | writes: is_context_valid, validation_score
  8. synthesizer        → reads: all_context       | writes: sub_task_answers
  9. answer_aggregator  → reads: sub_task_answers  | writes: final_answer
 10. summarizer         → reads: final_answer      | writes: global_summary, content_summary
"""

from typing import Annotated, Any, Dict, List, Literal
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """Shared mutable state passed between every node in the LangGraph graph.

    Fields annotated with ``Annotated[List[str], operator.add]`` use LangGraph's
    built-in list-reducer: nodes append items and the framework concatenates them
    automatically, so parallel branches never overwrite each other's results.
    """

def override(old, new):
    return new if new is not None else old


class AgentState(TypedDict):
    """Shared mutable state passed between every node in the LangGraph graph.

    Fields annotated with ``Annotated[List[str], operator.add]`` use LangGraph's
    built-in list-reducer: nodes append items and the framework concatenates them
    automatically, so parallel branches never overwrite each other's results.

    Other fields use the `override` reducer to prevent conflicts when parallel
    nodes return the same configuration or metadata keys.
    """

    # ------------------------------------------------------------------
    # Configuration — set per-request or from defaults
    # ------------------------------------------------------------------

    llm_provider: Annotated[str, override]
    """Provider to use for LLM calls (e.g., 'gemini', 'ollama')."""

    embedding_provider: Annotated[str, override]
    """Provider to use for embeddings (e.g., 'google', 'ollama')."""

    # ------------------------------------------------------------------
    # Input — set once by the API before the graph starts
    # ------------------------------------------------------------------

    question: Annotated[str, override]
    """Original user question received by the endpoint."""

    input_data: Annotated[str, override]
    """Raw document text / upload content passed to context_compressor at startup."""

    content_summary: Annotated[str, override]
    """Short textual index of the knowledge base; used by the router to decide
    whether the current sub-task can be answered from internal data (RAG) or
    requires a web search. Updated by the summarizer after each run."""

    # ------------------------------------------------------------------
    # Memory — per-user long-term context via Mem0
    # ------------------------------------------------------------------

    user_id: Annotated[str, override]
    """User identifier for per-user memory. Defaults to 'guest'."""

    user_memory: Annotated[str, override]
    """Facts retrieved from Mem0 about the user, injected into prompts."""

    # ------------------------------------------------------------------
    # Node 1 — Ambiguity Checker (AmbiguityCheckOutput)
    # ------------------------------------------------------------------

    is_ambiguous: Annotated[bool, override]
    """True → the question is too vague; the graph short-circuits and returns
    rejection_reason to the user without running the rest of the pipeline."""

    rejection_reason: Annotated[str, override]
    """Human-readable explanation returned to the user when is_ambiguous=True.
    Written by ambiguity_checker as 'rejection_reason' (maps to AmbiguityCheckOutput.reason)."""

    # ------------------------------------------------------------------
    # Node 2 — Planner (PlannerOutput)
    # ------------------------------------------------------------------

    sub_tasks: Annotated[List[str], override]
    """Ordered list of 1–5 independent sub-task strings decomposed from question."""

    current_task_index: Annotated[int, override]
    """Zero-based pointer into sub_tasks; incremented after each sub-task loop."""

    current_task: Annotated[str, override]
    """The sub-task string being processed in the current loop iteration."""

    # ------------------------------------------------------------------
    # Node 3 — Knowledge Router (KnowledgeRouterOutput)
    # ------------------------------------------------------------------

    route: Annotated[Literal["rag", "web", "llm_knowledge"], override]
    """Routing decision for current_task.
      "rag"           → retrieve from ChromaDB using HyDE
      "web"           → run a Tavily web search
      "llm_knowledge" → answer directly from model weights (no retrieval)
    """

    # ------------------------------------------------------------------
    # Node 4 — HyDE Generator
    # ------------------------------------------------------------------

    hyde_query: Annotated[str, override]
    """Hypothetical document text generated by the LLM to improve dense-retrieval
    recall. Embedded and used as the query vector against ChromaDB."""

    # ------------------------------------------------------------------
    # Node 5 — RAG Retriever / Web Searcher
    # ------------------------------------------------------------------

    rag_context: Annotated[str, override]
    """Concatenated text chunks retrieved from ChromaDB for current_task."""

    web_context: Annotated[str, override]
    """Concatenated snippets returned by Tavily for current_task."""

    # ------------------------------------------------------------------
    # Node 6 — Context Merger + Validator (ValidatorOutput)
    # ------------------------------------------------------------------

    all_context: Annotated[List[str], operator.add]
    """Accumulated context strings from all parallel subtask branches.
    Uses operator.add reducer so each subtask_runner branch *appends* its
    retrieved context instead of overwriting — critical for fan-in correctness."""

    is_context_valid: Annotated[bool, override]
    """True when validation_score >= 0.7 (ValidatorOutput.is_valid).
    If False, the graph may retry retrieval or proceed with a partial answer."""

    validation_score: Annotated[float, override]
    """Coverage score from 0.0 (nothing relevant found) to 1.0 (fully covered).
    Maps directly to ValidatorOutput.score."""

    # ------------------------------------------------------------------
    # Node 7 — Sub-task Synthesizer
    # Uses operator.add so parallel sub-task branches append without collision
    # ------------------------------------------------------------------

    sub_task_answers: Annotated[List[str], operator.add]
    """Per-sub-task answers accumulated across all loop iterations.
    The answer aggregator joins these into the final response."""

    # ------------------------------------------------------------------
    # Node 8 — Answer Aggregator (final output)
    # ------------------------------------------------------------------

    final_answer: Annotated[str, override]
    """Single synthesized answer returned to the caller after all sub-tasks complete."""

    # ------------------------------------------------------------------
    # Node 9 — Summarizer (updates knowledge base index)
    # ------------------------------------------------------------------

    global_summary: Annotated[Dict[str, Any], override]
    """Structured summary of the knowledge base produced by the summarizer node.
    Schema mirrors GLOBAL_SUMMARY_PROMPT output:
      {
        "topics":          List[str],
        "entities":        List[str],
        "time_range":      str,
        "summary":         str,
        "total_documents": int,
        "total_chunks":    int,
      }
    Used to refresh content_summary for the next session.
    """
