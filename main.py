from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from uuid import uuid4
from pdf_processor import extract_text_chunks
from vector_store import store_documents
from search_engine import get_embedding, query_chroma, query_gemini, extract_page_filter
from models import (
    ChatRequest, ChatResponse, 
    QuestionPaperRequest, QuestionPaperResponse, 
    AnswerSheetCorrectionResponse,OcrRequest
)
from questionpaper import generate_question_paper
from firestore11 import store_question_paper, get_question_paper,store_studentmarks

from ansheetcorrection import (
    run_ocr_sequential_internal,
    merge_ocr_results,
    correct_answers_single_rag
)
import chromadb
from typing import List
import base64
from google.cloud import firestore
from ocr import process_image
from fastapi.responses import JSONResponse



OCR_URL = "http://localhost:1234/ocr"

app = FastAPI(title="Flat Textbook RAG API")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    print("[main] /upload_pdf")
    if not file.filename.lower().endswith(".pdf"):
        print("[main] ERROR: non‑PDF")
        raise HTTPException(400, "Only PDF allowed")
    name = file.filename.rsplit(".", 1)[0].replace(" ", "_")
    pdf = await file.read()
    print(f"[main] Read {file.filename} ({len(pdf)} bytes)")
    chunks = extract_text_chunks(pdf)
    docs = [
        {"id": str(uuid4()), "metadata": {"page_no": c["page_no"], "text": c["text"]}}
        for c in chunks
    ]
    store_documents(name, docs)
    print(f"[main] Stored {len(docs)} chunks in '{name}'")
    return {"message": f"Stored {len(docs)} chunks in '{name}'."}

@app.post("/chat_with_textbook", response_model=ChatResponse)
async def chat_with_textbook(req: ChatRequest):
    print("[main] /chat_with_textbook")
    emb  = get_embedding(req.prompt)
    pf   = extract_page_filter(req.prompt)
    hits = query_chroma(req.collection_name, emb, page_filter=pf)
    context = "\n\n".join(f"(Page {h['metadata']['page_no']}): {h['text']}" for h in hits)
    ans     = query_gemini(req.prompt, context)
    print("[main] Returning answer + context")
    return {
        "answer": ans,
        "context_with_pages": [
            {"page_no": h["metadata"]["page_no"], "text": h["text"]} for h in hits
        ]
    }

@app.post("/generate_question_paper", response_model=QuestionPaperResponse)
async def create_question_paper(req: QuestionPaperRequest):
    print("[main] /generate_question_paper")
    try:
        result = generate_question_paper(
            collection_name=req.collection_name,
            user_prompt=req.user_prompt,
            paper_type=req.paper_type
        )
        if "error" in result:
            raise HTTPException(400, result["error"])
        print("[main] Question paper generated successfully")
        return result
    except Exception as e:
        print(f"[main] Error generating question paper: {e}")
        raise HTTPException(500, f"Failed to generate question paper: {str(e)}")

@app.post("/correct_answersheet", response_model=AnswerSheetCorrectionResponse)
async def correct_answersheet(
    images: List[UploadFile] = File(...),
    studentid: str = Form(...),
    questionpaperdocfromfiretore: str = Form(...),
    subject: str = Form(...),
    chromadbcollectionname: str = Form(...),
    correctiontype: str = Form(...)
):
    base64_images = [
        base64.b64encode(await img.read()).decode('utf-8')
        for img in images
    ]
    # Synchronous in-process OCR using the actual OCR ML function
    ocr_results = await run_ocr_sequential_internal(base64_images, process_image)
    merged_answers = merge_ocr_results(ocr_results)
    qp_doc = get_question_paper(questionpaperdocfromfiretore)
    if not qp_doc:
        raise HTTPException(404, f"No question paper found with ID: {questionpaperdocfromfiretore}")
    total, details = correct_answers_single_rag(
        merged_answers, qp_doc, chromadbcollectionname, correctiontype=correctiontype
    )
    resp = {
        "totalmarks": total,
        "eachquestion_marks": details,
        "studentid": studentid,
        "questionpaperdocfromfiretore": questionpaperdocfromfiretore,
        "subject": subject
    }
    store_studentmarks(resp)
    return resp

@app.get("/list_chromadb_collections")
def list_chromadb_collections():
    client = chromadb.PersistentClient(path="./chroma_db")

    collections = client.list_collections()
    try:
        collection_names = [col.name for col in collections]
    except AttributeError:
        collection_names = list(collections)
    return {"collections": collection_names}

@app.get("/list_questionpapers")
def list_questionpapers():
    client = firestore.Client()
    docs = client.collection("questionpaper").stream()
    papers = [doc.id for doc in docs]
    return {"questionpapers": papers}

# API endpoint
@app.post("/ocr")
async def execute_ocr(request: OcrRequest):
    decoded_bytes = base64.b64decode(request.base64)
    result = await process_image(decoded_bytes)
    return JSONResponse(content=result)
