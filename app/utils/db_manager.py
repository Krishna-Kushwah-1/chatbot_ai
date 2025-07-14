import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaDBManager:
    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = None
        
    def initialize_db(self):
        """Initialize or load existing Chroma DB"""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )
        return self.vectordb
    
    def store_documents(self, docs: List[Document]):
        """Store documents in Chroma DB with persistence"""
        if not docs:
            return None
        
        # Create new collection with documents
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.persist_dir
        )
        self.vectordb = vectordb
        return vectordb
    
    def get_retriever(self, k: int = 2, score_threshold: float = 0.6):
        """Get retriever with specified parameters"""
        if not self.vectordb:
            self.initialize_db()
            
        return self.vectordb.as_retriever(
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            }
        )
    
    def clear_database(self):
        """Completely clear the Chroma database"""
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectordb = None
        return True
    
    def get_collection_stats(self):
        """Get basic statistics about the collection"""
        if not self.vectordb:
            self.initialize_db()
            
        collection = self.vectordb._client.get_collection(self.vectordb._collection.name)
        return {
            "count": collection.count(),
            "metadata": collection.metadata
        }