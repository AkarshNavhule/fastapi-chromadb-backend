from google.cloud import firestore
import os
import re

# Optionally, set this if you're not setting it in your environment:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sahayak-d88d3-2e1f13a7b2bc.json"

client = firestore.Client()

def _get_next_doc_id(base_name: str, collection_ref) -> str:
    """
    Finds the next available document ID in the format 'base_name-1', 'base_name-2', etc.
    """
    existing_ids = [doc.id for doc in collection_ref.stream() if doc.id.startswith(base_name + "-")]
    max_index = 0
    for doc_id in existing_ids:
        try:
            suffix = int(doc_id.split("-")[-1])
            max_index = max(max_index, suffix)
        except ValueError:
            continue
    return f"{base_name}-{max_index + 1}"

def store_question_paper(response: dict) -> str:
    """
    Stores the question paper in Firestore with a doc ID like 'paper-1', 'paper-2', etc.
    """
    collection_ref = client.collection("questionpaper")
    base_name = "paper"  # Always use this as base prefix
    doc_id = _get_next_doc_id(base_name, collection_ref)

    firestore_doc = dict(response)
    firestore_doc["question_paper_id"] = doc_id
    if "question_paper" in firestore_doc:
        firestore_doc["question_paper"]["id"] = doc_id

    collection_ref.document(doc_id).set(firestore_doc)
    print(f"[firestore] Stored question paper: {doc_id}")
    return doc_id

def get_question_paper(doc_id: str) -> dict:
    """
    Fetch a specific question paper by its document ID.
    """
    collection_ref = client.collection("questionpaper")
    doc = collection_ref.document(doc_id).get()
    if doc.exists:
        return doc.to_dict()
    else:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    """
    Retrieves a question paper from Firestore by document ID.

    Args:
        doc_id (str): Firestore document ID (eg. ...-0001)

    Returns:
        dict: The document if it exists, else None
    """
    doc_ref = client.collection("questionpaper").document(doc_id)
    doc = doc_ref.get()
    if doc.exists:
        print(f"[firestore] Loaded question paper: {doc_id}")
        return doc.to_dict()
    else:
        print(f"[firestore] No document found for ID: {doc_id}")
        return None

def store_studentmarks(response: dict) -> str:
    """
    Stores the answer correction response in Firestore collection 'studentmarks'.
    Uses studentid as Firestore document ID.

    Args:
        response (dict): The full correction API response.

    Returns:
        str: Firestore document ID
    """
    collection_ref = client.collection("studentmarks")
    studentid = str(response.get("studentid"))
    if not studentid:
        raise ValueError("Response must include 'studentid'")
    doc_id = "studentid-"+studentid  

    firestore_doc = dict(response)
    collection_ref.document(doc_id).set(firestore_doc)
    print(f"[firestore] Stored student marks for: {doc_id}")
    return doc_id


def get_studentmarks(studentid: str) -> dict:
    doc_ref = client.collection("studentmarks").document(studentid)
    doc = doc_ref.get()
    if doc.exists:
        print(f"[firestore] Loaded student marks: {studentid}")
        return doc.to_dict()
    else:
        print(f"[firestore] No document found for ID: {studentid}")
        return None
