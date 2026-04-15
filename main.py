import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from src.core.config import settings
from src.utils.logger import logger
from src.core.orchestrator import IngestionOrchestrator
from src.llm.embeddings import EmbeddingFactory
from src.retrieval.vector_db import VectorDBManager
from src.agents.graph import graph
from src.llm.factory import LLLMFactory

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Hierarchical RAG API",
    description="Hierarchical Agentic RAG system supporting multi-platform (Gemini & Ollama)",
    version="1.1.0",
)

@app.on_event("startup")
async def startup_event():
    """Initialize default LlamaIndex settings on startup."""
    try:
        LLLMFactory.configure_llama_index_settings()
        logger.info("Default LlamaIndex settings configured.")
    except Exception as e:
        logger.error(f"Failed to set default LlamaIndex settings: {e}")

# --- Models ---
class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = "guest"
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    input_data_summary: Optional[str] = ""

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_count: int
    collection: str
    summary: str
    provider: str

# --- Endpoints ---

@app.get("/")
def root():
    return {
        "message": "Hierarchical RAG API is running",
        "project": settings.app.project_name,
        "version": settings.app.version,
        "default_provider": settings.llm.provider,
    }

@app.get("/health")
def health_check():
    """Check API health status."""
    return {"status": "healthy", "version": settings.app.version}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    embedding_provider: Optional[str] = Query(None, description="google or ollama"),
):
    """Upload PDF, extract text, chunk, and index into hierarchical vector DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    logger.info(f"Starting document ingestion: {file.filename}")

    # 1. Lưu file tạm thời
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
    # 2. Use Orchestrator for processing
        orchestrator = IngestionOrchestrator(provider=embedding_provider)
        result = await orchestrator.ingest_pdf(tmp_path, embedding_provider=embedding_provider)
        
        return IngestResponse(**result)

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

@app.post("/chat")
async def chat(request: ChatRequest):
    """Send query to Agentic Hierarchical RAG pipeline."""
    # Initialize settings for this session
    llm_provider = request.llm_provider or settings.llm.provider
    embedding_provider = request.embedding_provider or settings.llm.provider
    
    LLLMFactory.configure_llama_index_settings(provider=llm_provider)
    
    logger.info(f"Received question: {request.question} (LLM: {llm_provider})")

    input_data = request.input_data_summary or ""
    content_summary = ""

    # If no input_data_summary, try loading from storage
    if not input_data:
        factory = EmbeddingFactory()
        embedding_model = factory.get_embedding(provider=embedding_provider)
        db_manager = VectorDBManager(embedding_model=embedding_model, provider=embedding_provider)

        content_summary = db_manager.get_summary(settings.system_paths.collection_name)
        if content_summary:
            logger.info(f"Loaded summary from storage for {embedding_provider}")

    # Initialize graph state
    initial_state = {
        "question": request.question,
        "user_id": request.user_id or "guest",
        "input_data": input_data,
        "content_summary": content_summary,
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "sub_task_answers": [],
        "all_context": [],
        "current_task_index": 0,
    }

    try:
        # Execute workflow graph
        result = await graph.ainvoke(initial_state)

        return {
            "question": request.question,
            "answer": result.get("final_answer"),
            "rejection_reason": result.get("rejection_reason"),
            "is_ambiguous": result.get("is_ambiguous", False),
            "sub_tasks": result.get("sub_tasks", []),
            "meta": {
                "llm": llm_provider,
                "embedding": embedding_provider,
                "user_id": request.user_id or "guest",
                "engine": "LlamaIndex AutoMerging"
            },
        }
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
async def reset_database(
    embedding_provider: Optional[str] = Query(None, description="google or ollama")
):
    """Clear vector database for a specific provider."""
    provider = embedding_provider or settings.llm.provider
    factory = EmbeddingFactory()
    embeddings = factory.get_embedding(provider=provider) 
    db_manager = VectorDBManager(embedding_model=embeddings, provider=provider)
    
    if db_manager.reset_db():
        return {"status": "success", "message": f"Data for {provider} has been cleared."}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear data.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
