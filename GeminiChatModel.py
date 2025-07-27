import re
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GENERATION_MODEL, EMBEDDING_MODEL, TOP_K
from vector_store import get_or_create_collection
import os

print("[leaderboard_chat] Configuring Gemini client…")
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))  # reads GEMINI_API_KEY from env

def get_embedding(text: str) -> list[float]:
    """Generate embedding for query text."""
    print("[leaderboard_chat] Generating embedding…")
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return resp.embeddings[0].values 

def query_gemini(prompt: str, context: str, system_prompt: str) -> str:
    """Generate answer using Gemini with context and system prompt."""
    print("[leaderboard_chat] Generating answer with Gemini…")
    
    cfg = types.GenerateContentConfig(system_instruction=system_prompt)
    combined = f"Context (Student Leaderboard Data):\n{context}\n\nQuestion:\n{prompt}"
    
    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        config=cfg,
        contents=combined
    )
    print("[leaderboard_chat] Answer received")
    return resp.text

def search_leaderboard(query: str, collection_name: str = "student_leaderboard") -> list:
    """Search the leaderboard collection for relevant documents."""
    print(f"[leaderboard_chat] Searching collection '{collection_name}'…")
    
    try:
        # Get the ChromaDB collection
        collection = get_or_create_collection(collection_name)
        
        # Generate embedding for the query
        query_embedding = get_embedding(query)
        
        # Search for similar documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"[leaderboard_chat] Found {len(results['documents'][0])} relevant documents")
        return results
    
    except Exception as e:
        print(f"Error searching leaderboard: {e}")
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

def format_context(search_results: dict) -> str:
    """Format search results into context for the LLM."""
    if not search_results['documents'][0]:
        return "No relevant student data found."
    
    context_parts = []
    documents = search_results['documents'][0]
    metadatas = search_results['metadatas'][0]
    distances = search_results['distances'][0]
    
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        context_part = f"Student Record {i+1} (Relevance Score: {1-distance:.3f}):\n"
        context_part += f"Document Content: {doc}\n"
        
        if metadata:
            context_part += "Additional Details:\n"
            for key, value in metadata.items():
                context_part += f"  - {key}: {value}\n"
        
        context_parts.append(context_part)
    
    return "\n" + "="*50 + "\n".join(context_parts)

def leaderboard_chat(user_query: str, system_prompt: str = None) -> str:
    """Main function to handle leaderboard chat queries."""
    
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """You are a helpful educational assistant with access to student leaderboard data. 
        You can answer questions about:
        - Student rankings and positions
        - Individual student scores in different subjects
        - Student feedback and comments
        - Performance comparisons between students
        - Subject-wise analysis
        
        Always provide accurate information based on the context provided. If specific information 
        is not available in the context, clearly state that. Be friendly and encouraging when 
        discussing student performance."""
    
    print(f"[leaderboard_chat] Processing query: '{user_query}'")
    
    # Search for relevant documents
    search_results = search_leaderboard(user_query)
    
    # Format context from search results
    context = format_context(search_results)
    
    # Generate response using Gemini
    response = query_gemini(user_query, context, system_prompt)
    
    return response

async def interactive_chat(chat_prompt):
    """Interactive chat interface for leaderboard queries."""
    print("🎓 Student Leaderboard Chat Assistant")
    print("="*50)
    print("Ask me anything about student rankings, scores, or feedback!")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Custom system prompt for interactive mode
    system_prompt = """You are a friendly educational assistant helping users explore student leaderboard data.
    
    You can help with queries like:
    - "Who is ranked 1st?" or "Who is in first place?"
    - "How much did [student name] score in [subject]?"
    - "What's the feedback for [student name]?"
    - "Compare [student1] and [student2] performance"
    - "Show me top 5 students in [subject]"
    
    Always be encouraging and positive when discussing student performance. If you can't find 
    specific information, suggest alternative ways to phrase the question."""
    
    while True:
        try:
            # user_input = input("🤔 Ask me: ").strip()
            
            # if user_input.lower() in ['quit', 'exit', 'bye']:
            #     print("📚 Thanks for using the Leaderboard Chat! Goodbye!")
            #     break
            
            # if not user_input:
            #     print("Please enter a question about the student leaderboard.")
            #     continue
            
            print("\n🔍 Searching leaderboard data...")
            response = leaderboard_chat(chat_prompt, system_prompt)
            
            print(f"\n📊 Answer: {response}\n")
            print("-" * 50)
            return response
            
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try rephrasing your question.\n")
            return {"error": str(e), "message": "An error occurred while processing your request. Please try again."}

# Example usage function
def run_examples():
    """Run some example queries to demonstrate functionality."""
    examples = [
        "Who is ranked 1st?",
        "How much did Raju score in Kannada subject?",
        "What's the feedback for Ramesh?",
        "Show me the top 3 students",
        "Which student scored highest in Mathematics?"
    ]
    
    print("🎓 Running Example Queries:")
    print("="*50)
    
    for query in examples:
        print(f"\n🤔 Query: {query}")
        print("🔍 Processing...")
        
        try:
            response = leaderboard_chat(query)
            print(f"📊 Answer: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    # You can choose to run interactive chat or examples
    choice = input("Choose mode:\n1. Interactive Chat\n2. Run Examples\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_chat()
    elif choice == "2":
        run_examples()
    else:
        print("Invalid choice. Starting interactive chat...")
        interactive_chat()
