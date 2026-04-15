"""Routing helpers and retrieval tool nodes for the Hierarchical RAG graph.

This module contains:
1. Router functions — conditional edge callbacks for LangGraph.
2. Retrieval tool nodes — work nodes that fetch information from RAG, Web, or LLM knowledge.
"""

from langgraph.graph import END
from langgraph.constants import Send
from typing import List

from llama_index.core.retrievers import AutoMergingRetriever
from langchain_community.tools.tavily_search import TavilySearchResults

from src.agents.state import AgentState
from src.core.config import settings
from src.llm.embeddings import EmbeddingFactory
from src.llm.factory import LLLMFactory
from src.retrieval.vector_db import VectorDBManager
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Singleton instances & Helpers
# ---------------------------------------------------------------------------

_indices = {}  # Cache: {provider_name: LlamaIndexInstance}
_tavily = None       # Lazy initialized


def get_index(embedding_provider: str = None):
    """Get or load a LlamaIndex index instance for a specific provider."""
    if embedding_provider is None:
        embedding_provider = settings.llm.provider.lower()
    
    embedding_provider = embedding_provider.lower()

    if embedding_provider not in _indices:
        logger.info(f"Loading Index for provider: {embedding_provider}")
        # Ensure global settings match the provider before loading/creating index
        LLLMFactory.configure_llama_index_settings(provider=embedding_provider)
        
        factory = EmbeddingFactory()
        embeddings = factory.get_embedding(provider=embedding_provider)

        db_manager = VectorDBManager(embedding_model=embeddings, provider=embedding_provider)
        index = db_manager.get_index(collection_name=settings.system_paths.collection_name)
        
        if index:
            _indices[embedding_provider] = index
        else:
            return None

    return _indices[embedding_provider]


def get_tavily():
    """Lazy initialize Tavily search tool."""
    global _tavily
    if _tavily is None:
        _tavily = TavilySearchResults(
            max_results=settings.retrieval.top_k,
            tavily_api_key=settings.tavily_api_key,
        )
    return _tavily


def _format_subtask_report(current_task: str, result: str, source: str) -> str:
    """Wrap a retrieval result in a structured, LLM-friendly report block."""
    return (
        f"\n=== SUB-TASK REPORT ===\n"
        f"SUB-TASK   : {current_task}\n"
        f"SOURCE     : {source}\n"
        f"FINDINGS   : {result}\n"
        f"======================="
    )


# ---------------------------------------------------------------------------
# Router functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_ambiguity(state: AgentState) -> str:
    if state.get("is_ambiguous", False):
        return "rejection_handler"
    return "planner"


def route_after_router(state: AgentState) -> str:
    route = state.get("route", "web")
    if route == "rag":
        return "hyde"
    elif route == "web":
        return "web_searcher"
    return "llm_node"


def route_after_validator(state: AgentState) -> str:
    if state.get("is_context_valid", False):
        return END
    return "web_searcher"


def fan_out_subtasks(state: AgentState) -> List[Send]:
    return [
        Send("subtask_runner", {**state, "current_task": task})
        for task in state.get("sub_tasks", [])
    ]


# ---------------------------------------------------------------------------
# Tool nodes
# ---------------------------------------------------------------------------

def rag_retriever(state: AgentState) -> dict:
    """Retrieve chunks from ChromaDB using LlamaIndex AutoMergingRetriever."""
    query = state.get("hyde_query") or state.get("current_task", "")
    current_task = state.get("current_task", "")
    embedding_provider = state.get("embedding_provider")

    logger.info(f"[RAG] Provider: {embedding_provider} | Query: {query[:50]}...")

    try:
        index = get_index(embedding_provider)
        if not index:
            raise ValueError(f"Index not found for {embedding_provider}")

        # Initialize AutoMergingRetriever
        # It searches leaf nodes and automatically merges them into parent nodes if threshold met
        base_retriever = index.as_retriever(similarity_top_k=settings.retrieval.top_k)
        retriever = AutoMergingRetriever(
            base_retriever, 
            index.storage_context, 
            verbose=True
        )

        nodes = retriever.retrieve(query)
        retrieved_text = "\n\n".join(node.get_content() for node in nodes) if nodes else ""
        
        logger.info(f"[RAG] Retrieved {len(nodes)} nodes (merged if applicable)")

    except Exception as exc:
        logger.error(f"[RAG] Retrieval failed: {exc}")
        retrieved_text = ""

    report = _format_subtask_report(
        current_task=current_task,
        result=retrieved_text or "(không tìm thấy dữ liệu liên quan)",
        source=f"Hierarchical RAG ({embedding_provider or 'default'})",
    )
    return {
        "all_context": [retrieved_text],
        "sub_task_answers": [report],
    }


def web_searcher(state: AgentState) -> dict:
    """Run a Tavily web search."""
    current_task = state.get("current_task", "")
    logger.info(f"[Web] Query: {current_task[:50]}...")

    try:
        search = get_tavily()
        results = search.invoke({"query": current_task})
        snippets = [r.get("content", "") for r in results if isinstance(r, dict)]
        search_text = "\n\n".join(s for s in snippets if s)
    except Exception as exc:
        logger.error(f"[Web] Search failed: {exc}")
        search_text = ""

    report = _format_subtask_report(
        current_task=current_task,
        result=search_text or "(không tìm thấy kết quả từ web)",
        source="Web Search",
    )
    return {
        "all_context": [search_text],
        "sub_task_answers": [report],
    }


def llm_node(state: AgentState) -> dict:
    """Answer from LLM internal knowledge. Supports dynamic LLM provider."""
    current_task = state.get("current_task", "")
    llm_provider = state.get("llm_provider")

    logger.info(f"[LLM] Provider: {llm_provider} | Task: {current_task[:50]}...")

    try:
        client = LLLMFactory.create_client("rag", provider=llm_provider)
        prompt = f"Trả lời câu hỏi sau dựa trên kiến thức của bạn:\n\n{current_task}"
        response = client.get_llm().invoke(prompt)
        llm_text = response.content
    except Exception as exc:
        logger.error(f"[LLM] Call failed: {exc}")
        llm_text = ""

    report = _format_subtask_report(
        current_task=current_task,
        result=llm_text or "(LLM không thể tạo câu trả lời)",
        source=f"LLM Knowledge ({llm_provider or 'default'})",
    )
    return {
        "all_context": [llm_text],
        "sub_task_answers": [report],
    }
