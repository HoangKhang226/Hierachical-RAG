from src.core.config import settings
from src.utils.logger import logger
from langchain_chroma import Chroma
import json
from pathlib import Path


class VectorDBManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.persist_directory = settings.system_paths.vector_db
        self._vector_store = None
        self.metadata_path = Path(self.persist_directory).parent / "collection_metadata.json"

    def _init_db(self, collection_name: str):
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory,
            )
        logger.info(f"Đã kết nối tới VectorDB tại: {self.persist_directory}")

    def add_documents(
        self, documents: list, collection_name: str = "default_collection"
    ):
        self._init_db(collection_name)
        if not documents:
            logger.warning("Danh sách tài liệu trống, không có gì để lưu.")
            return []

        try:
            ids = self._vector_store.add_documents(documents)
            logger.info(f"Đã lưu {len(ids)} chunks vào hệ thống.")
            return ids
        except Exception as e:
            logger.error(f"Lỗi hệ thống khi lưu vào DB: {e}")
            return []

    def get_retriever(self, collection_name: str = "default_collection", k: int = 5):
        self._init_db(collection_name)
        return self._vector_store.as_retriever(search_kwargs={"k": k})

    def delete_collection(self, collection_name: str = "hierarchical_rag"):
        """Delete the collection and clear the local vector store instance."""
        try:
            # Re-init if not already initialized
            self._init_db(collection_name)
            
            if self._vector_store:
                self._vector_store.delete_collection()
                self._vector_store = None
                logger.info(f"Đã xóa sạch dữ liệu trong collection: {collection_name}")
        except Exception as e:
            logger.error(f"Lỗi khi xóa collection: {e}")

    def reset_db(self):
        """Physically delete the vector database directory and summary metadata."""
        import shutil
        try:
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)
                logger.info(f"Đã xóa thư mục VectorDB: {self.persist_directory}")
            
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                logger.info(f"Đã xóa file metadata: {self.metadata_path}")
            
            self._vector_store = None
            return True
        except Exception as e:
            logger.error(f"Lỗi khi reset VectorDB: {e}")
            return False
            
    def save_summary(self, collection_name: str, summary: str):
        """Save a summary for a specific collection to a JSON file."""
        data = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        data[collection_name] = summary
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Đã lưu tóm tắt cho bộ sưu tập: {collection_name}")

    def get_summary(self, collection_name: str) -> str:
        """Retrieve the summary for a specific collection."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get(collection_name, "")
        return ""
