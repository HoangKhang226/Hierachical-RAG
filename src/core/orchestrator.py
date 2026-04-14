from pathlib import Path
from typing import Optional, List, Tuple

from src.core.config import settings
from src.utils.logger import logger
from src.processors.pdf_engine import PDFEngine
from src.processors.chunker import Chunker
from src.llm.embeddings import EmbeddingFactory
from src.retrieval.vector_db import VectorDBManager
from src.llm.factory import LLLMFactory
from src.prompt.template import CONTEXT_COMPRESSION_PROMPT


class IngestionOrchestrator:
    """Orchestrate document ingestion: Extract -> Chunk -> Embed -> Index."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.llm.provider
        self.pdf_engine = PDFEngine()
        self.chunker = Chunker()
        self.embedding_factory = EmbeddingFactory()
        
        # Ensure LlamaIndex is configured for this provider
        LLLMFactory.configure_llama_index_settings(provider=self.provider)

    async def generate_summary(self, nodes: List, provider: str) -> str:
        """Generate a short summary from the first document nodes."""
        if not nodes:
            return ""

        # Take first 10 nodes for overview summary
        sample_text = "\n\n".join([n.get_content() for n in nodes[:10]])

        client = LLLMFactory.create_client("summary", provider=provider)
        prompt = CONTEXT_COMPRESSION_PROMPT.format(input_data=sample_text)

        try:
            # Use invoke for stability if provider async support is limited
            response = client.get_llm().invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating automatic summary: {e}")
            return "Document summary unavailable."

    async def ingest_pdf(self, pdf_path: Path, embedding_provider: Optional[str] = None) -> dict:
        """Run the full PDF ingestion pipeline."""
        provider = embedding_provider or self.provider
        
        # 1. Extract text from PDF
        doc = self.pdf_engine.process_pdf(pdf_path)
        if doc.metadata.get("status") == "error":
            raise Exception(f"PDF processing failed: {doc.metadata.get('error')}")

        # 2. Hierarchical Chunking
        all_nodes, leaf_nodes = self.chunker.chunk(doc)

        # 3. Initialize Vector DB Manager
        embedding_model = self.embedding_factory.get_embedding(provider=provider)
        db_manager = VectorDBManager(embedding_model=embedding_model, provider=provider)
        
        # 4. Index into vector database
        ids = db_manager.add_documents(
            all_nodes, collection_name=settings.system_paths.collection_name
        )

        # 5. Generate and store metadata summary
        summary = await self.generate_summary(all_nodes, self.provider)
        db_manager.save_summary(settings.system_paths.collection_name, summary)

        return {
            "status": "success",
            "filename": pdf_path.name,
            "chunks_count": len(ids),
            "collection": settings.system_paths.collection_name,
            "summary": summary,
            "provider": provider
        }
