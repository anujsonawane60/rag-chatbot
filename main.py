import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cohere
from pinecone import Pinecone, ServerlessSpec
import uvicorn
import PyPDF2
import io
import docx
import re
import tempfile
import shutil
import time
from datetime import datetime
from typing import List, Dict
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Multi-RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Config:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    DIMENSION = 1024
    CHUNK_SIZE = 500

    @classmethod
    def validate_env_vars(cls):
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found")
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found")

class ChatbotManager:
    def __init__(self):
        self.chatbots = {}
        self.base_upload_dir = "uploaded_files"
        self.chat_history_dir = "chat_history"
        os.makedirs(self.base_upload_dir, exist_ok=True)
        os.makedirs(self.chat_history_dir, exist_ok=True)
        self.load_existing_chatbots()

    def load_existing_chatbots(self):
        if os.path.exists(self.base_upload_dir):
            for chatbot_name in os.listdir(self.base_upload_dir):
                if os.path.isdir(os.path.join(self.base_upload_dir, chatbot_name)):
                    self.initialize_existing_chatbot(chatbot_name)

    def initialize_existing_chatbot(self, chatbot_name: str):
        index_name = f"rag-chatbot-{chatbot_name.lower()}"
        service_manager = ServiceManager()
        service_manager.initialize_services(index_name)
        
        chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
        files = os.listdir(chatbot_dir)
        
        self.chatbots[chatbot_name] = {
            "index_name": index_name,
            "service_manager": service_manager,
            "files": files,
            "created_date": self.get_creation_date(chatbot_dir)
        }

    def get_creation_date(self, directory: str) -> str:
        timestamp = os.path.getctime(directory)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def create_chatbot(self, chatbot_name: str):
        try:
            if not chatbot_name or not re.match("^[a-zA-Z0-9-_]+$", chatbot_name):
                raise HTTPException(400, "Invalid chatbot name. Use only letters, numbers, hyphens and underscores")

            if chatbot_name in self.chatbots:
                raise HTTPException(400, "Chatbot with this name already exists")

            chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
            os.makedirs(chatbot_dir, exist_ok=True)

            index_name = f"rag-chatbot-{chatbot_name.lower()}"[:62]
            service_manager = ServiceManager()
            
            try:
                service_manager.initialize_services(index_name)
            except Exception as e:
                if os.path.exists(chatbot_dir):
                    shutil.rmtree(chatbot_dir)
                raise HTTPException(500, f"Failed to initialize services: {str(e)}")

            self.chatbots[chatbot_name] = {
                "index_name": index_name,
                "service_manager": service_manager,
                "files": [],
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return {
                "status": "success",
                "message": f"Chatbot '{chatbot_name}' created successfully",
                "name": chatbot_name
            }
        except Exception as e:
            chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
            if os.path.exists(chatbot_dir):
                shutil.rmtree(chatbot_dir)
            if chatbot_name in self.chatbots:
                del self.chatbots[chatbot_name]
            raise HTTPException(500, f"Error creating chatbot: {str(e)}")

    def delete_chatbot(self, chatbot_name: str):
        if chatbot_name not in self.chatbots:
            raise HTTPException(404, "Chatbot not found")

        service_manager = self.chatbots[chatbot_name]["service_manager"]
        service_manager.pinecone_client.delete_index(
            self.chatbots[chatbot_name]["index_name"]
        )

        chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
        if os.path.exists(chatbot_dir):
            shutil.rmtree(chatbot_dir)

        history_file = os.path.join(self.chat_history_dir, f"{chatbot_name}.json")
        if os.path.exists(history_file):
            os.remove(history_file)

        del self.chatbots[chatbot_name]

        return {"status": "success", "message": f"Chatbot '{chatbot_name}' deleted"}

    def get_chatbot_info(self, chatbot_name: str) -> Dict:
        if chatbot_name not in self.chatbots:
            raise HTTPException(404, "Chatbot not found")

        return {
            "name": chatbot_name,
            "files": self.chatbots[chatbot_name]["files"],
            "created_date": self.chatbots[chatbot_name]["created_date"]
        }

    def save_chat_history(self, chatbot_name: str, query: str, answer: str):
        history_file = os.path.join(self.chat_history_dir, f"{chatbot_name}.json")
        
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": answer
        })
        
        with open(history_file, 'w') as f:
            json.dump(history, f)

    def get_chat_history(self, chatbot_name: str) -> List[Dict]:
        history_file = os.path.join(self.chat_history_dir, f"{chatbot_name}.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
class ServiceManager:
    def __init__(self):
        self.cohere_client = None
        self.pinecone_client = None
        self.index = None

    def initialize_services(self, index_name: str):
        try:
            # Initialize Cohere
            if not Config.COHERE_API_KEY:
                raise ValueError("COHERE_API_KEY not found in environment variables")
            self.cohere_client = cohere.Client(Config.COHERE_API_KEY)

            # Initialize Pinecone
            if not Config.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            self.pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)

            # Check if index exists and create if it doesn't
            existing_indexes = self.pinecone_client.list_indexes().names()
            
            if index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {index_name}")
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=Config.DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                time.sleep(20)
            
            # Initialize index
            self.index = self.pinecone_client.Index(index_name)
            
            # Verify index is accessible
            try:
                self.index.describe_index_stats()
            except Exception as e:
                raise Exception(f"Failed to access index: {str(e)}")

        except Exception as e:
            raise Exception(f"Service initialization error: {str(e)}")

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
            print(f"PDF extraction error: {str(e)}")
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
            print(f"DOCX extraction error: {str(e)}")
            return ""

    @staticmethod
    def chunk_text(text, chunk_size=Config.CHUNK_SIZE):
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

# Initialize managers
Config.validate_env_vars()
chatbot_manager = ChatbotManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/chatbots")
async def list_chatbots():
    chatbots_info = []
    for name in chatbot_manager.chatbots:
        chatbots_info.append(chatbot_manager.get_chatbot_info(name))
    return {"status": "success", "chatbots": chatbots_info}

@app.post("/chatbot/create")
async def create_chatbot(request: Request):
    try:
        data = await request.json()
        chatbot_name = data.get("name")
        
        if not chatbot_name:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Chatbot name is required"}
            )
        
        result = chatbot_manager.create_chatbot(chatbot_name)
        return JSONResponse(content=result)
        
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": str(he.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )

@app.delete("/chatbot/{chatbot_name}")
async def delete_chatbot(chatbot_name: str):
    return chatbot_manager.delete_chatbot(chatbot_name)

@app.post("/chatbot/{chatbot_name}/upload")
async def upload_file(chatbot_name: str, file: UploadFile = File(...)):
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")

    try:
        # Validate file size (optional, adjust max_size as needed)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        file_content = b''
        
        # Read file in chunks to check size
        while chunk := await file.read(8192):
            file_size += len(chunk)
            file_content += chunk
            if file_size > max_size:
                raise HTTPException(400, "File too large (max 10MB)")

        # Save file
        file_path = os.path.join(chatbot_manager.base_upload_dir, chatbot_name, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Process file
        if file.filename.lower().endswith('.pdf'):
            text = TextProcessor.extract_text_from_pdf(file_content)
        elif file.filename.lower().endswith('.docx'):
            text = TextProcessor.extract_text_from_docx(file_content)
        elif file.filename.lower().endswith('.txt'):
            text = file_content.decode('utf-8')
        else:
            raise HTTPException(400, "Unsupported file format")

        if not text:
            raise HTTPException(400, "No text could be extracted from the file")

        # Process chunks
        chunks = TextProcessor.chunk_text(text)
        service_manager = chatbot_manager.chatbots[chatbot_name]["service_manager"]

        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
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

        # Update chatbot files list
        chatbot_manager.chatbots[chatbot_name]["files"].append(file.filename)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"File '{file.filename}' uploaded successfully",
                "details": {
                    "filename": file.filename,
                    "size": file_size,
                    "chunks_processed": len(vectors_to_upsert),
                    "text_length": len(text)
                }
            }
        )

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": str(he.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Upload failed: {str(e)}"}
        )


@app.post("/chatbot/{chatbot_name}/ask")
async def ask_question(chatbot_name: str, request: Request):
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")

    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            raise HTTPException(400, "Query required")

        service_manager = chatbot_manager.chatbots[chatbot_name]["service_manager"]

        # Generate query embedding
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
            return {"status": "error", "answer": "No relevant information found"}

        # Generate answer
        prompt = f"""Context: {context}

Question: {query}

Please provide a clear and concise answer based on the context above."""

        response = service_manager.cohere_client.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )

        answer = response.generations[0].text.strip()
        print("Answer:", answer)

        # Save to chat history
        chatbot_manager.save_chat_history(chatbot_name, query, answer)

        return {
            "status": "success",
            "answer": answer,
            "context": relevant_chunks
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/chatbot/{chatbot_name}/history")
async def get_chat_history(chatbot_name: str):
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")
    
    history = chatbot_manager.get_chat_history(chatbot_name)
    return {"status": "success", "history": history}

if __name__ == "__main__":
    try:
        # Validate environment and create necessary directories
        Config.validate_env_vars()
        os.makedirs("uploaded_files", exist_ok=True)
        os.makedirs("chat_history", exist_ok=True)
        
        # Start the application
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
