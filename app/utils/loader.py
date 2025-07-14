import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)

def load_and_split(path: str):
    """Load documents and split into chunks."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext == ".csv":
        loader = CSVLoader(file_path=path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)