from app.utils.db_manager import ChromaDBManager

# Initialize the DB manager
db_manager = ChromaDBManager()

def load_retriever(persist_dir: str = None, k: int = 2, score_threshold: float = 0.6):
    """Load retriever using the DB manager"""
    if persist_dir:
        db_manager.persist_dir = persist_dir
    return db_manager.get_retriever(k=k, score_threshold=score_threshold)