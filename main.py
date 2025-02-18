import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cohere
from pinecone import Pinecone, ServerlessSpec
import uvicorn
import PyPDF2
import io
import docx
import re
import tempfile
import shutil
import numpy as np
import time

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Config:
    # API Keys
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Constants
    DIMENSION = 1024  # Cohere's embed-english-v3.0 dimension
    INDEX_NAME = "rag-chatbot"
    CHUNK_SIZE = 500

    @classmethod
    def validate_env_vars(cls):
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

class ServiceManager:
    def _init_(self):
        self.cohere_client = None
        self.pinecone_client = None
        self.index = None

    def initialize_services(self):
        try:
            # Initialize Cohere
            self.cohere_client = cohere.Client(Config.COHERE_API_KEY)
            print("Cohere client initialized successfully")

            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)
            print("Pinecone initialized successfully")

            # Delete existing index if it exists
            if Config.INDEX_NAME in self.pinecone_client.list_indexes().names():
                self.pinecone_client.delete_index(Config.INDEX_NAME)
                print(f"Deleted existing index: {Config.INDEX_NAME}")
                # Wait for the index to be fully deleted
                time.sleep(20)

            # Create new index
            self.pinecone_client.create_index(
                name=Config.INDEX_NAME,
                dimension=Config.DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Created new Pinecone index: {Config.INDEX_NAME}")
            
            # Wait for the index to be ready
            time.sleep(10)
            
            # Get the index
            self.index = self.pinecone_client.Index(Config.INDEX_NAME)
            print("Index initialization complete")

        except Exception as e:
            print(f"Service initialization error: {str(e)}")
            raise

class TextProcessor:
    @staticmethod
    def extract_text_from_pdf(file_bytes):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_bytes):
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return ""

    @staticmethod
    def chunk_text(text, chunk_size=Config.CHUNK_SIZE):
        try:
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return [chunk for chunk in chunks if chunk.strip()]
        except Exception as e:
            print(f"Error chunking text: {str(e)}")
            return []

# Initialize services
Config.validate_env_vars()
service_manager = ServiceManager()
service_manager.initialize_services()

async def process_file_upload(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        if file.filename.lower().endswith('.pdf'):
            with open(temp_path, 'rb') as f:
                text = TextProcessor.extract_text_from_pdf(f.read())
        elif file.filename.lower().endswith('.docx'):
            with open(temp_path, 'rb') as f:
                text = TextProcessor.extract_text_from_docx(f.read())
        elif file.filename.lower().endswith('.txt'):
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise HTTPException(400, "Unsupported file format")
    finally:
        os.unlink(temp_path)

    return text

async def process_chunks(chunks):
    try:
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                # Use Cohere's embedding model with correct parameters
                response = service_manager.cohere_client.embed(
                    texts=[chunk],
                    model='embed-english-v3.0',
                    input_type="search_document"
                )
                embedding = response.embeddings[0]

                vectors_to_upsert.append({
                    'id': f'chunk_{i}_{os.urandom(4).hex()}',
                    'values': embedding,
                    'metadata': {'text': chunk}
                })

        if vectors_to_upsert:
            service_manager.index.upsert(vectors=vectors_to_upsert)
            print(f"Successfully uploaded {len(vectors_to_upsert)} vectors")

        return len(vectors_to_upsert)
    except Exception as e:
        print(f"Error processing chunks: {str(e)}")
        return 0

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        text = await process_file_upload(file)

        if not text:
            raise HTTPException(400, "No text could be extracted from the file")

        chunks = TextProcessor.chunk_text(text)
        if not chunks:
            raise HTTPException(400, "No valid text chunks were created")

        chunks_processed = await process_chunks(chunks)

        if chunks_processed == 0:
            raise HTTPException(500, "Failed to process and store text chunks")

        return JSONResponse(content={
            "status": "success",
            "message": f"File processed successfully. {chunks_processed} chunks stored.",
            "filename": file.filename
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "details": "Error occurred during file processing"
            }
        )

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        query = data.get("query")

        if not query:
            raise HTTPException(400, "Query parameter is required")

        # Generate query embedding using Cohere with correct parameters
        response = service_manager.cohere_client.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type="search_query"
        )
        query_embedding = response.embeddings[0]

        # Search in Pinecone
        search_results = service_manager.index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        relevant_chunks = [match['metadata']['text'] for match in search_results['matches']]
        context = " ".join(relevant_chunks)

        if not context.strip():
            return JSONResponse(content={
                "status": "error",
                "answer": "No relevant information found in documents",
                "context": []
            })

        # Generate answer using Cohere
        prompt = f"""Context: {context}

Question: {query}

Please provide a clear and concise answer based on the context above. If the context doesn't contain relevant information, say so."""

        response = service_manager.cohere_client.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )

        answer = response.generations[0].text.strip()

        return JSONResponse(content={
            "status": "success",
            "answer": answer,
            "context": relevant_chunks
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "details": "Error occurred during question processing"
            }
        )

if __name__ == "_main_":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)