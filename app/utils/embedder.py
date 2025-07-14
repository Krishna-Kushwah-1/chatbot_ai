from app.utils.db_manager import ChromaDBManager

# Initialize the DB manager
db_manager = ChromaDBManager()

def store_documents(docs, persist_dir: str = None):
    """Store documents using the DB manager"""
    if persist_dir:
        db_manager.persist_dir = persist_dir
    return db_manager.store_documents(docs)