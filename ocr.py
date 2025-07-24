import base64
import re
from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
from google.oauth2 import service_account
from models import OcrRequest

# Auth
credentials = service_account.Credentials.from_service_account_file(
    "visionJson.json",
    scopes=["https://www.googleapis.com/auth/cloud-vision"]
)

client = vision.ImageAnnotatorClient(credentials=credentials)

# Text extraction utility
async def extract_answers(ocr_text: str) -> dict:
    lines = ocr_text.split('\n')
    answers = {}
    current_q = None
    current_answer = []

    question_pattern = re.compile(r'^(?:Q(?:uestion)?\.?\s*)?(\d+)[\.\)]?\s*')

    for line in lines:
        q_match = question_pattern.match(line.strip())
        if q_match:
            if current_q is not None:
                answers[current_q] = ' '.join(current_answer).strip()
                current_answer = []
            current_q = q_match.group(1)
            line = question_pattern.sub('', line).strip()
            if line:
                current_answer.append(line)
        else:
            if current_q and line.strip():
                current_answer.append(line.strip())
    
    if current_q:
        answers[current_q] = ' '.join(current_answer).strip()
    return answers

# OCR logic
async def process_image(byte_array: bytes) -> dict:
    image = vision.Image(content=byte_array)
    response: AnnotateImageResponse = client.document_text_detection(
        image=image,
        image_context={"language_hints": ["en-t-i0-handwrit"]}
    )

    annotation = response.full_text_annotation
    full_text = ""
    for page in annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                paragraph_text = ''
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    paragraph_text += word_text + ' '
                full_text += "\n" + paragraph_text

    return await extract_answers(full_text)

