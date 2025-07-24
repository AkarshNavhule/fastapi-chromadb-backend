from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4

class ChatRequest(BaseModel):
    prompt: str
    collection_name: str

class ChatResponse(BaseModel):
    answer: str
    context_with_pages: list[dict]

class QuestionPaperRequest(BaseModel):
    collection_name: str
    user_prompt: str
    paper_type: str = "medium"  # easy, medium, hard

class QuestionPaperResponse(BaseModel):
    question_paper: Dict[str, Any]
    sources: List[Dict[str, Any]]
    error: Optional[str] = None

class AnswerSheetCorrectionRequest(BaseModel):
    studentid: str
    images: List[bytes]  # FastAPI: Use List[UploadFile] in endpoint
    questionpaperdocfromfiretore: str
    subject: str
    chromadbcollectionname: str
    correctiontype: str  # "easy", "medium", or "hard"

class EachQuestionMark(BaseModel):
    question_no: str
    question: str
    studentanswer: str
    marks: int
    chromadbsource: str
    remarks: str

class AnswerSheetCorrectionResponse(BaseModel):
    totalmarks: str  # e.g., "45/50"
    eachquestion_marks: List[EachQuestionMark]
    studentid: str
    questionpaperdocfromfiretore: str
    subject: str

class OcrRequest(BaseModel):
    base64: str