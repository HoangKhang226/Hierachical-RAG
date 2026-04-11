from src.core.config import settings
from src.utils.logger import logger
from src.processors.pdf_engine import PDFEngine
from typing import Literal
from docling.chunking import HierarchicalChunker
from docling_core.types.doc.document import DoclingDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunker:
    """Splits document text into retrieval-ready chunks.

    Supports two strategies:
      - "docling": Markdown-aware splitting that preserves heading hierarchy,
        followed by a secondary recursive split to enforce size limits.
      - "pypdf": Hierarchical parent/child splitting for plain text, attaching
        a parent index to each child chunk for context retrieval.
    """

    def chunk(self, purpose: Literal["docling", "pypdf"], docs: str):
        """Split a document string into a list of LangChain Document chunks.

        Args:
            purpose: Chunking strategy to use — "docling" or "pypdf".
            docs: Raw document text to be split.

        Returns:
            A list of LangChain Document objects representing the final chunks.
        """
        if purpose == "docling":
            # Split on Markdown headers first to keep section context intact
            headers_to_split_on = [
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]
            chunker = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            chunks = chunker.split_text(docs)

            chunk_size = settings.chunking.child_chunk_size
            chunk_overlap = settings.chunking.child_chunk_overlap

            # Secondary split to cap chunk size within the configured limit
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            final_chunks = text_splitter.split_documents(chunks)
            logger.info(f"Markdown chunking completed — {len(final_chunks)} chunks produced")
            return final_chunks

        else:
            parent_chunk_size = settings.chunking.parent_chunk_size
            parent_chunk_overlap = settings.chunking.parent_chunk_overlap

            child_chunk_size = settings.chunking.child_chunk_size
            child_chunk_overlap = settings.chunking.parent_chunk_overlap

            parent_chunker = RecursiveCharacterTextSplitter(
                chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
            )
            docs = [Document(page_content=docs)]
            parent_chunks = parent_chunker.split_documents(docs)
            logger.debug(f"Parent chunking produced {len(parent_chunks)} chunks")

            child_chunker = RecursiveCharacterTextSplitter(
                chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
            )
            child_chunks = []
            for i, doc in enumerate(parent_chunks):
                # Tag each parent chunk with its index for later context retrieval
                doc.metadata["Doc"] = i
                child_chunks.extend(child_chunker.split_documents([doc]))

            logger.info(
                f"Hierarchical chunking completed — {len(parent_chunks)} parents, "
                f"{len(child_chunks)} children"
            )
            return child_chunks
