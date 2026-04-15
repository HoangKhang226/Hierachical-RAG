from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from src.agents.state import AgentState
from src.prompt.template import (
    CONTEXT_COMPRESSION_PROMPT,
    AMBIGUITY_CHECK_PROMPT,
    PLANNER_PROMPT,
    KNOWLEDGE_ROUTER_PROMPT,
    HYDE_PROMPT,
    VALIDATOR_PROMPT,
    GLOBAL_SUMMARY_PROMPT,
    SYNTHESIZER_PROMPT,
    USER_MEMORY_SECTION,
)
from src.llm.factory import LLLMFactory
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Output Schemas (Pydantic) — used only for LLM structured output parsing
# Node functions extract fields from these and return plain dicts.
# ---------------------------------------------------------------------------


class AmbiguityCheckOutput(BaseModel):
    """Schema for Node 1 — Ambiguity Checker."""

    is_ambiguous: bool = Field(
        description="True if the question is ambiguous or lacks reference info, False if clear."
    )
    reason: str = Field(
        default="",
        description="Explain why the question is judged as ambiguous or clear based on rules.",
    )


class PlannerOutput(BaseModel):
    """Schema for Node 2 — Planner."""

    sub_tasks: List[str] = Field(
        description="List of 1 to 5 sub-tasks decomposed from the original question.",
    )


class KnowledgeRouterOutput(BaseModel):
    """Schema for Node 3 — Knowledge Router."""

    route: Literal["rag", "web", "llm_knowledge"] = Field(
        description="Most suitable data source for the task (rag, web, or llm_knowledge)."
    )


class ValidatorOutput(BaseModel):
    """Schema for Node 6 — Validator."""

    score: float = Field(
        description="Information coverage score from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
    is_valid: bool = Field(
        description="True if score >= 0.7 (PASS), False otherwise."
    )
    reason: str = Field(
        description="Detailed analysis: identify key entities and their presence in context."
    )


# ---------------------------------------------------------------------------
# Node functions — MUST return dict with exact AgentState field names
# ---------------------------------------------------------------------------


def _build_memory_section(state: AgentState) -> str:
    """Return a formatted user-memory block with Vietnamese fallback if empty."""
    mem = state.get("user_memory", "")
    content = mem if mem else "Chưa có dữ liệu thông tin về người dùng."
    return USER_MEMORY_SECTION.format(user_memory=content)



def context_compressor(state: AgentState) -> dict:
    """Node 0: Compress raw input into a short content_summary.

    Reads:  input_data, llm_provider
    Writes: content_summary
    """
    # If a summary already exists (e.g. loaded from persistence), skip re-summarizing
    if state.get("content_summary"):
        logger.info("Using pre-existing content_summary, skipping compression.")
        return {}

    if not state.get("input_data"):
        logger.warning("No input_data provided for compression, returning empty summary.")
        return {"content_summary": ""}

    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("summary", provider=provider)
    
    prompt = CONTEXT_COMPRESSION_PROMPT.format(input_data=state["input_data"])
    response = client.get_llm().invoke(prompt)
    return {"content_summary": response.content}


def ambiguity_checker(state: AgentState) -> dict:
    """Node 1: Decide whether the question is too vague to process.

    Reads:  question, content_summary, user_memory, llm_provider
    Writes: is_ambiguous, rejection_reason
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("rag", provider=provider)
    
    # Handle empty content_summary with fallback
    summary = state.get("content_summary") or "Chưa có dữ liệu nội dung tài liệu."
    
    prompt = AMBIGUITY_CHECK_PROMPT.format(
        question=state["question"],
        content_summary=summary,
        user_memory_section=_build_memory_section(state),
    )
    structured_llm = client.get_structed_llm(AmbiguityCheckOutput)
    result: AmbiguityCheckOutput = structured_llm.invoke(prompt)
    return {
        "is_ambiguous": result.is_ambiguous,
        "rejection_reason": result.reason,
    }


def planner(state: AgentState) -> dict:
    """Node 2: Decompose the question into an ordered list of sub-tasks.

    Reads:  question, llm_provider
    Writes: sub_tasks
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("rag", provider=provider)
    
    prompt = PLANNER_PROMPT.format(
        question=state["question"],
        user_memory_section=_build_memory_section(state),
    )
    structured_llm = client.get_structed_llm(PlannerOutput)
    result: PlannerOutput = structured_llm.invoke(prompt)
    return {"sub_tasks": result.sub_tasks}


def rejection_handler(state: AgentState) -> dict:
    """Node: Handle ambiguous questions by mapping rejection_reason to final_answer.
    
    This ensures the pipeline reaches the update_memory node with a valid response.
    """
    reason = state.get("rejection_reason", "Tôi không chắc chắn về yêu cầu của bạn.")
    logger.info(f"[RejectionHandler] Mapping rejection reason to final answer.")
    return {"final_answer": reason}


def knowledge_router(state: AgentState) -> dict:
    """Node 3: Route the current sub-task to the right retrieval path.

    Reads:  current_task, content_summary, llm_provider
    Writes: route
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("classifier", provider=provider)
    
    prompt = KNOWLEDGE_ROUTER_PROMPT.format(
        current_task=state["current_task"],
        global_summary=state["content_summary"],
    )
    structured_llm = client.get_structed_llm(KnowledgeRouterOutput)
    result: KnowledgeRouterOutput = structured_llm.invoke(prompt)
    return {"route": result.route}


def hyde(state: AgentState) -> dict:
    """Node 4: Generate a hypothetical document to improve RAG recall (HyDE).

    Reads:  current_task, llm_provider
    Writes: hyde_query
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("rag", provider=provider)
    
    prompt = HYDE_PROMPT.format(current_task=state["current_task"])
    response = client.get_llm().invoke(prompt)
    return {"hyde_query": response.content}


def validator(state: AgentState) -> dict:
    """Node 6: Score context coverage and decide if retrieval was sufficient.

    Reads:  current_task, all_context, llm_provider
    Writes: is_context_valid, validation_score
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("classifier", provider=provider)
    
    prompt = VALIDATOR_PROMPT.format(
        current_task=state["current_task"],
        all_context="\n\n".join(state.get("all_context", [])),
    )
    structured_llm = client.get_structed_llm(ValidatorOutput)
    result: ValidatorOutput = structured_llm.invoke(prompt)
    return {
        "is_context_valid": result.score >= 0.7,
        "validation_score": result.score,
    }


def global_summary(state: AgentState) -> dict:
    """Node 8: Produce a structured summary of the knowledge base.

    Reads:  content_summary, sub_task_answers, llm_provider
    Writes: global_summary
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("summary", provider=provider)
    
    # Prepare inputs for the prompt
    sub_tasks_context = "\n\n".join(state.get("sub_task_answers", []))
    if not sub_tasks_context:
        sub_tasks_context = "(No specific sub-task findings available)"

    prompt = GLOBAL_SUMMARY_PROMPT.format(
        documents=sub_tasks_context
    )
    response = client.get_llm().invoke(prompt)
    return {"global_summary": response.content}


def synthesizer(state: AgentState) -> dict:
    """Node 9: Synthesize all labelled sub-task reports into the final answer.

    Reads:  sub_task_answers, global_summary, question, llm_provider
    Writes: final_answer
    """
    provider = state.get("llm_provider")
    client = LLLMFactory.create_client("summary", provider=provider)
    
    merged_reports = "\n".join(state.get("sub_task_answers", []))
    prompt = SYNTHESIZER_PROMPT.format(
        question=state.get("question", ""),
        all_context=merged_reports,
        user_memory_section=_build_memory_section(state),
    )
    response = client.get_llm().invoke(prompt)
    return {"final_answer": response.content}

