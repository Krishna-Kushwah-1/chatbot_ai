import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from ollama import Client
import time
from typing import List

from app.utils.loader import load_and_split
from app.utils.embedder import store_documents
from app.utils.retriever import load_retriever
from app.utils.db_manager import ChromaDBManager

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = ChromaDBManager()
UPLOAD_DIR = "./data/raw_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    try:
        db_manager.initialize_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Couldn't initialize database: {e}")

@app.post("/upload_knowledge")
async def upload_knowledge(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    report = []
    all_docs = []
    
    for file in files:
        dest_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            docs = load_and_split(dest_path)
            if docs:
                all_docs.extend(docs)
                report.append(f"✅ {file.filename}: {len(docs)} chunks")
            else:
                report.append(f"⚠️ {file.filename}: No text found")
        except Exception as e:
            report.append(f"❌ {file.filename}: {str(e)}")
    
    if all_docs:
        try:
            store_documents(all_docs)
            report.insert(0, f"📚 Total chunks: {len(all_docs)}")
            stats = db_manager.get_collection_stats()
            report.append(f"📊 Database now contains {stats['count']} vectors")
        except Exception as e:
            report.append(f"Error storing documents: {str(e)}")
    
    return JSONResponse({
        "message": f"Processed {len(files)} file(s)",
        "details": report
    })

@app.post("/query")
async def query(question: str = Form(...)):
    try:
        retriever = load_retriever()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Knowledge base error: {str(e)}")
    
    start_time = time.time()
    
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        context_time = time.time()
        print(f"Retrieval time: {context_time - start_time:.2f}s")
    except Exception as e:
        context = ""
        print(f"Retrieval error: {e}")
    
    client = Client()
    prompt = (
        "Answer the question concisely based ONLY on this context. "
        "If unsure, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    def generate():
        try:
            stream = client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={
                    "num_predict": 250,
                    "temperature": 0.4
                }
            )
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"Error generating answer: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Response-Time": f"{time.time() - start_time:.2f}s"}
    )

@app.get("/db_stats")
async def get_db_stats():
    try:
        stats = db_manager.get_collection_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_db")
async def reset_db():
    try:
        db_manager.clear_database()
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))