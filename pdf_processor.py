import fitz  # PyMuPDF

def extract_text_chunks(pdf_bytes: bytes, chunk_size: int = 1000):
    print("[pdf_processor] Extracting text...")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for idx in range(len(doc)):
        page_no = idx + 1
        text = doc[idx].get_text()
        print(f"[pdf_processor] Page {page_no} length: {len(text)} chars")
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                chunks.append({"page_no": page_no, "text": chunk})
    print(f"[pdf_processor] Created {len(chunks)} chunks")
    return chunks
