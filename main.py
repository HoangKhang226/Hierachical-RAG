import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from src.core.config import settings
from src.utils.logger import logger
from src.processors.pdf_engine import PDFEngine
from src.processors.chunker import Chunker
from src.llm.embeddings import EmbeddingFactory
from src.retrieval.vector_db import VectorDBManager
from src.agents.graph import graph
from src.llm.factory import LLLMFactory
from src.prompt.template import CONTEXT_COMPRESSION_PROMPT

app = FastAPI(
    title="Hierarchical RAG API",
    description="Agentic RAG system with multi-provider support (Gemini & Ollama)",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    input_data_summary: Optional[str] = ""


class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_count: int
    collection: str
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def generate_initial_summary(chunks: list, llm_provider: str) -> str:
    """Generate a concise summary from the first few chunks of the document."""
    if not chunks:
        return ""

    # Take first 5 chunks to build a global perspective
    sample_text = "\n\n".join([c.page_content for c in chunks[:]])

    client = LLLMFactory.create_client("summary", provider=llm_provider)
    prompt = CONTEXT_COMPRESSION_PROMPT.format(input_data=sample_text)

    try:
        response = await client.get_llm().ainvoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating initial summary: {e}")
        return "Tóm tắt tài liệu không khả dụng."


@app.get("/")
def root():
    return {
        "message": "Hierarchical RAG API is running",
        "provider": settings.llm.provider,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    embedding_provider: Optional[str] = Query(None, description="google or ollama"),
    strategy: Optional[str] = Query("docling", description="docling or pypdf"),
):
    """Upload a PDF, extract text, chunk and index into the vector database."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    provider = embedding_provider or settings.llm.provider
    logger.info(f"Starting ingestion for {file.filename} using {provider} embeddings")

    # 1. Save uploaded file to temp
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # 2. Process PDF
        engine = PDFEngine()
        extraction_result = engine.process_pdf(tmp_path)

        if extraction_result.get("status") == "error":
            raise HTTPException(status_code=500, detail="PDF processing failed.")

        content = extraction_result["content"]
        metadata_source = extraction_result["metadata"]

        # 3. Chunking
        chunker = Chunker()
        # use the extraction source to guide chunking strategy if pypdf was used
        chunks = chunker.chunk(
            purpose=metadata_source if strategy == "pypdf" else strategy, docs=content
        )

        # 4. Vector DB Ingestion
        factory = EmbeddingFactory()
        embedding_model = factory.get_embedding(provider=provider)

        db_manager = VectorDBManager(embedding_model=embedding_model)
        ids = db_manager.add_documents(
            chunks, collection_name=settings.system_paths.collection_name
        )

        # 5. Generate and save summary
        summary = await generate_initial_summary(chunks, settings.llm.provider)
        db_manager.save_summary(settings.system_paths.collection_name, summary)

        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_count=len(ids),
            collection=settings.system_paths.collection_name,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/chat")
async def chat(request: ChatRequest):
    """Submit a question to the agentic Hierarchical RAG pipeline."""
    logger.info(f"Received question: {request.question}")

    # Auto-load summary if not provided
    input_data = request.input_data_summary or ""
    content_summary = ""

    if not input_data:
        # Try to load saved summary from DB
        factory = EmbeddingFactory()
        provider = request.embedding_provider or settings.llm.provider
        embedding_model = factory.get_embedding(provider=provider)
        db_manager = VectorDBManager(embedding_model=embedding_model)

        content_summary = db_manager.get_summary(settings.system_paths.collection_name)
        if content_summary:
            logger.info(
                f"Loaded persistent summary for {settings.system_paths.collection_name}"
            )

    # Initialize state
    initial_state = {
        "question": request.question,
        "input_data": input_data,
        "content_summary": content_summary,
        "llm_provider": request.llm_provider or settings.llm.provider,
        "embedding_provider": request.embedding_provider or settings.llm.provider,
        "sub_task_answers": [],
        "all_context": [],
        "current_task_index": 0,
    }

    try:
        # Run graph
        # Note: In a production app, you might want to use graph.astream for real-time progress
        result = await graph.ainvoke(initial_state)

        return {
            "question": request.question,
            "answer": result.get("final_answer"),
            "rejection_reason": result.get("rejection_reason"),
            "is_ambiguous": result.get("is_ambiguous", False),
            "sub_tasks": result.get("sub_tasks", []),
            "meta": {
                "llm": request.llm_provider or settings.llm.provider,
                "embedding": request.embedding_provider or settings.llm.provider,
            },
        }
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_collection(
    collection_name: Optional[str] = Query(None),
    embedding_provider: Optional[str] = Query(None)
):
    """Wipe a specific collection in the vector database."""
    name = collection_name or settings.system_paths.collection_name
    provider = embedding_provider or settings.llm.provider
    
    factory = EmbeddingFactory()
    embeddings = factory.get_embedding(provider=provider)
    db_manager = VectorDBManager(embedding_model=embeddings)
    
    db_manager.delete_collection(collection_name=name)
    return {"status": "success", "message": f"Collection '{name}' deleted."}


@app.delete("/reset")
async def reset_database():
    """Physically delete the entire vector database directory and all metadata.
    
    Use this if you encounter dimensionality mismatch errors or want a clean slate.
    """
    # We can use any embedding model for reset since it just deletes files
    factory = EmbeddingFactory()
    embeddings = factory.get_embedding(provider="ollama") 
    db_manager = VectorDBManager(embedding_model=embeddings)
    
    success = db_manager.reset_db()
    if success:
        return {"status": "success", "message": "Vector database directory physically deleted."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reset database directory.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
