from fastapi import FastAPI
from supabase import create_client
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pypdf import PdfReader
import os
import glob

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI()

# Embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


@app.get("/")
def home():
    return {"message": "Mining Safety AI Backend Running"}


@app.get("/process-pdfs")
def process_pdfs():

    pdf_files = glob.glob("pdfs/*.pdf")

    for pdf_path in pdf_files:

        reader = PdfReader(pdf_path)

        full_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                full_text += text

        # File name as title
        title = os.path.basename(pdf_path)

        # Check if PDF already processed
        existing = supabase.table("documents").select("*").eq("title", title).execute()

        if existing.data:
            continue

        # Save document info
        document = supabase.table("documents").insert({
            "title": title,
            "department": title.split("_")[0]
        }).execute()

        document_id = document.data[0]["id"]

        # Split into chunks
        chunk_size = 500

        chunks = [
            full_text[i:i + chunk_size]
            for i in range(0, len(full_text), chunk_size)
        ]

        # Store chunks + embeddings
        for chunk in chunks:

            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )

            embedding = embedding_response.data[0].embedding

            supabase.table("document_chunks").insert({
                "document_id": document_id,
                "chunk_text": chunk,
                "embedding": embedding
            }).execute()

    return {
        "message": "All PDFs processed successfully",
        "total_pdfs": len(pdf_files)
    }


@app.get("/ask")
def ask_question(question: str):

    # Convert question into embedding
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_embedding = embedding_response.data[0].embedding

    # Search relevant chunks
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 10
        }
    ).execute()

    matches = response.data

    # Build context from retrieved chunks
    context = ""

    for match in matches:
        context += match["chunk_text"] + "\n\n"

    # Prompt
    prompt = f"""
You are a mining safety assistant.

Answer ONLY from the provided safety documents.

If answer is not available in documents, say:
'Information not found in uploaded safety documents.'

Safety Documents:
{context}

User Question:
{question}
"""

    # OpenAI response
    ai_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a mining safety expert."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = ai_response.choices[0].message.content

    return {
        "question": question,
        "answer": answer
    }