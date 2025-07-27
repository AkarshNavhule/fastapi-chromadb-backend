from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from uuid import uuid4
from pdf_processor import extract_text_chunks
from vector_store import store_documents
from search_engine import get_embedding, query_chroma, query_gemini, extract_page_filter,query_gemini_ppt
from models import (
    ChatRequest, ChatResponse, 
    QuestionPaperRequest, QuestionPaperResponse, 
    AnswerSheetCorrectionResponse,OcrRequest, LeaderboardChatRequest
)
from questionpaper import generate_question_paper
from firestore11 import store_question_paper, get_question_paper,store_studentmarks
from StudentLeaderboardVectorStore import fetch_firestore_collection, upload_to_chroma
from GeminiChatModel import interactive_chat

from ansheetcorrection import (
    run_ocr_sequential_internal,
    merge_ocr_results,
    correct_answers_single_rag
)
import os
import boto3
import chromadb
from typing import List
import base64
from google.cloud import firestore 
from google.cloud import storage
from ocr import process_image
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
import asyncio
import aioboto3
app = FastAPI(title="Flat Textbook RAG API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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
    assignmentid: str = Form(...),
    classgrade: str = Form(...),
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
        "subject": subject,
        "assignmentid": assignmentid
        ,"classgrade": classgrade
    }
    # process_assessment_response(resp)
    # student_performance_resp= {"totalmarks": total,
        # "studentid": studentid,
        # "questionpaperdocfromfiretore": questionpaperdocfromfiretore,
        # "subject": subject,
        # "assignmentid": assignmentid
        # ,"classgrade": classgrade,
        # "createdtimestamp": firestore.SERVER_TIMESTAMP,}
    # student_performance(student_performance_resp)

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
@app.post("/create_ppt", response_model=ChatResponse)
async def create_ppt(req: ChatRequest):
    print("[main] /create_ppt")
    emb  = get_embedding(req.prompt)
    pf   = extract_page_filter(req.prompt)
    hits = query_chroma(req.collection_name, emb, page_filter=pf)
    context = "\n\n".join(f"(Page {h['metadata']['page_no']}): {h['text']}" for h in hits)
    ans     = query_gemini_ppt(req.prompt, context)
    print("[main] Returning answer + context")
    return {
        "answer": ans,
        "context_with_pages": [
            {"page_no": h["metadata"]["page_no"], "text": h["text"]} for h in hits
        ]
    }


@app.get("/upload_leaderboard_vector")
async def upload_leaderboard_vector():
    firestore_docs = fetch_firestore_collection("student_leaderboard")
    
    if not firestore_docs:
        print("No documents found in Firestore collection. Exiting.")
        return
    
    # Upload to ChromaDB
    success = upload_to_chroma(firestore_docs)
    
    if success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed!")
        return {"message": "Migration failed!"}
    return {"message": "Upload process completed."}

@app.post("/chat_with_leaderboard")
async def chat_with_leaderboard(req: LeaderboardChatRequest):
    resp = {"answer": await interactive_chat(req.prompt)}
    return resp



# rekognition = boto3.client(
#     'rekognition',
#     aws_access_key_id='AKIARLPXLUEWWLRIZFW4',
#     aws_secret_access_key='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',
#     region_name='us-east-1'
# )

AWS_CONFIG = {
    'aws_access_key_id': 'AKIARLPXLUEWWLRIZFW4',
    'aws_secret_access_key': '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',
    'region_name': 'us-east-1'
}

# Google Cloud Storage client
storage_client = storage.Client()
BUCKET_NAME = "studentimages1" 

async def download_single_blob_async(blob):
    """Async download of a single blob"""
    try:
        student_name = os.path.splitext(blob.name)[0]
        # Run the blocking download in thread pool
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(
            None, 
            blob.download_as_bytes
        )
        print(f"   • downloaded {blob.name} → {student_name}")
        return student_name, image_bytes
    except Exception as e:
        print(f"   ⚠️  Error downloading {blob.name}: {e}")
        return None, None

async def get_student_images_from_bucket_async() -> dict:
    """
    Async version: Download all student images from GCS bucket concurrently
    """
    try:
        print("📥  Fetching student list from GCS bucket …")
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Get list of blobs (this is still sync, but fast)
        blobs = list(bucket.list_blobs())
        image_blobs = [
            blob for blob in blobs 
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        if not image_blobs:
            print("   No image files found in bucket")
            return {}
        
        print(f"📁  Found {len(image_blobs)} images, downloading concurrently …")
        
        # Download all images concurrently
        download_tasks = [
            download_single_blob_async(blob) 
            for blob in image_blobs
        ]
        
        # Wait for all downloads to complete
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Filter out failed downloads and create student dict
        student_images = {}
        successful_downloads = 0
        
        for result in results:
            if isinstance(result, tuple) and result[0] and result[1]:
                student_name, image_bytes = result
                student_images[student_name] = image_bytes
                successful_downloads += 1
            elif isinstance(result, Exception):
                print(f"   ⚠️  Download exception: {result}")
        
        print(f"✅  Successfully downloaded {successful_downloads}/{len(image_blobs)} images\n")
        return student_images

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS access error: {e}")

async def check_student(group_img: bytes,
                        student_name: str,
                        student_img: bytes,
                        threshold: int = 80) -> tuple:
    """
    Async call to Rekognition → returns (name, "present"/"absent")
    """
    print(f"🔍  Comparing {student_name} …")
    session = aioboto3.Session()
    async with session.client("rekognition", **AWS_CONFIG) as rek:
        try:
            result = await rek.compare_faces(
                SourceImage = {"Bytes": student_img},
                TargetImage = {"Bytes": group_img},
                SimilarityThreshold = threshold
            )
            status = "present" if result["FaceMatches"] else "absent"
            print(f"   → {student_name}: {status}")
            return student_name, status
        except Exception as e:
            print(f"   ⚠️  {student_name}: error → {e}")
            return student_name, "absent"

# ─── API endpoint ───────────────────────────────────────────────────────────────
@app.post("/attendance")
async def mark_attendance(targetimage: UploadFile = File(...)):
    """
    Upload a group photo; returns {"alice": "present", "bob": "absent", …}
    Now with fully async bucket fetching!
    """
    print("\n===== New /attendance request =====")
    
    # Start group image read and bucket fetch concurrently
    print("🚀  Starting concurrent operations: group image read + bucket fetch")
    
    group_read_task = targetimage.read()
    bucket_fetch_task = get_student_images_from_bucket_async()
    
    # Wait for both operations to complete
    group_bytes, students = await asyncio.gather(
        group_read_task,
        bucket_fetch_task
    )
    
    print(f"👥  Group image: {targetimage.filename} ({len(group_bytes)//1024} KB)")
    
    if not students:
        raise HTTPException(status_code=404, detail="No student images in bucket")

    print(f"🎯  Starting face recognition for {len(students)} students …")
    
    # Kick off concurrent Rekognition calls
    recognition_tasks = [
        check_student(group_bytes, name, img)
        for name, img in students.items()
    ]
    results = await asyncio.gather(*recognition_tasks)

    attendance = dict(results)
    print(f"📊  Final attendance: {attendance}\n")
    return attendance

@app.get("/students")
async def list_students():
    """Get list of all students in the bucket"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs()
        
        students = []
        for blob in blobs:
            if blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                student_name = os.path.splitext(blob.name)[0]
                students.append(student_name)
        
        return {"students": students, "total": len(students)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing students: {str(e)}") 
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sahayak-d88d3-2e1f13a7b2bc.json"    
db = firestore.Client()
@app.get("/leaderboard")
async def get_leaderboard():
    students = db.collection('student_leaderboard').order_by('rank').stream()
    return [student.to_dict() for student in students]
