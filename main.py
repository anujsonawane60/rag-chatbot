import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cohere
from pinecone import Pinecone, ServerlessSpec
import uvicorn
from pypdf import PdfReader
import io
import asyncio
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

# Add CORS middleware (frontend is served from the same origin;
# add your deployed domain here when hosting the UI elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
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
    INDEX_NAME = "multi-rag-chatbot"  # one shared index; each chatbot gets its own namespace
    CHUNK_SIZE = 1500                 # chars (~375 tokens), well under the embed model's 512-token limit
    CHUNK_OVERLAP = 200               # chars carried across chunk boundaries to preserve context
    EMBED_BATCH_SIZE = 96             # Cohere embed API max texts per call
    EMBED_MODEL = "embed-english-v3.0"
    RERANK_MODEL = "rerank-v3.5"
    CHAT_MODEL = "command-a-03-2025"
    RETRIEVE_TOP_K = 20               # wide candidate set for the reranker
    RERANK_TOP_N = 4                  # final chunks passed to the LLM

    @classmethod
    def validate_env_vars(cls):
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found")
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found")

class ChatbotManager:
    def __init__(self, service_manager: "ServiceManager"):
        self.chatbots = {}
        self.service_manager = service_manager
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
        chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
        files = os.listdir(chatbot_dir)

        self.chatbots[chatbot_name] = {
            "namespace": chatbot_name,
            "files": files,
            "created_date": self.get_creation_date(chatbot_dir)
        }

    def get_creation_date(self, directory: str) -> str:
        timestamp = os.path.getctime(directory)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def create_chatbot(self, chatbot_name: str):
        if not chatbot_name or not re.match("^[a-zA-Z0-9-_]+$", chatbot_name):
            raise HTTPException(400, "Invalid chatbot name. Use only letters, numbers, hyphens and underscores")

        if chatbot_name in self.chatbots:
            raise HTTPException(400, "Chatbot with this name already exists")

        chatbot_dir = os.path.join(self.base_upload_dir, chatbot_name)
        os.makedirs(chatbot_dir, exist_ok=True)

        # No per-chatbot index anymore — vectors live in a namespace of the
        # shared index, which is created lazily on the first upsert
        self.chatbots[chatbot_name] = {
            "namespace": chatbot_name,
            "files": [],
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return {
            "status": "success",
            "message": f"Chatbot '{chatbot_name}' created successfully",
            "name": chatbot_name
        }

    def delete_chatbot(self, chatbot_name: str):
        if chatbot_name not in self.chatbots:
            raise HTTPException(404, "Chatbot not found")

        # Remove this chatbot's vectors (its namespace in the shared index)
        try:
            self.service_manager.index.delete(
                delete_all=True,
                namespace=self.chatbots[chatbot_name]["namespace"]
            )
        except Exception:
            pass  # namespace doesn't exist until the first upload

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
                while not self.pinecone_client.describe_index(index_name).status['ready']:
                    time.sleep(1)
            
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
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"
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
    def chunk_text(text, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Hard-split any single sentence longer than chunk_size so no chunk
        # exceeds the embedding model's context window
        split_sentences = []
        for sentence in sentences:
            while len(sentence) > chunk_size:
                split_sentences.append(sentence[:chunk_size])
                sentence = sentence[chunk_size:]
            if sentence:
                split_sentences.append(sentence)

        chunks = []
        current = []
        current_len = 0

        for sentence in split_sentences:
            if current and current_len + len(sentence) > chunk_size:
                chunks.append(" ".join(current).strip())
                # Carry trailing sentences forward as overlap so context
                # isn't lost at chunk boundaries
                tail = []
                tail_len = 0
                for s in reversed(current):
                    if tail_len + len(s) > overlap:
                        break
                    tail.insert(0, s)
                    tail_len += len(s) + 1
                current = tail
                current_len = tail_len
            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if chunk.strip()]

# Initialize managers — one shared index for all chatbots (namespaces keep
# them separated and avoid Pinecone's free-tier index limit)
Config.validate_env_vars()
service_manager = ServiceManager()
service_manager.initialize_services(Config.INDEX_NAME)
chatbot_manager = ChatbotManager(service_manager)

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
        
        result = await asyncio.to_thread(chatbot_manager.create_chatbot, chatbot_name)
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
    return await asyncio.to_thread(chatbot_manager.delete_chatbot, chatbot_name)

@app.post("/chatbot/{chatbot_name}/upload")
async def upload_file(chatbot_name: str, file: UploadFile = File(...)):
    if chatbot_name not in chatbot_manager.chatbots:
        raise HTTPException(404, "Chatbot not found")

    try:
        # Sanitize filename to prevent path traversal
        safe_filename = os.path.basename(file.filename or "").replace("\\", "")
        if not safe_filename or not re.match(r'^[\w\-. ()]+$', safe_filename):
            raise HTTPException(400, "Invalid filename. Use only letters, numbers, spaces, dots, hyphens and underscores")

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
        file_path = os.path.join(chatbot_manager.base_upload_dir, chatbot_name, safe_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        # Process file
        if safe_filename.lower().endswith('.pdf'):
            text = TextProcessor.extract_text_from_pdf(file_content)
        elif safe_filename.lower().endswith('.docx'):
            text = TextProcessor.extract_text_from_docx(file_content)
        elif safe_filename.lower().endswith('.txt'):
            text = file_content.decode('utf-8')
        else:
            raise HTTPException(400, "Unsupported file format")

        if not text:
            raise HTTPException(400, "No text could be extracted from the file")

        # Process chunks
        chunks = TextProcessor.chunk_text(text)
        service_manager = chatbot_manager.service_manager
        namespace = chatbot_manager.chatbots[chatbot_name]["namespace"]

        def embed_and_upsert():
            index = service_manager.index

            # Remove vectors from any previous upload of this file (IDs are
            # prefixed with the filename, so re-uploads don't duplicate chunks)
            try:
                for ids in index.list(prefix=f"{safe_filename}#", namespace=namespace):
                    index.delete(ids=list(ids), namespace=namespace)
            except Exception:
                pass  # namespace doesn't exist until the first upload

            # Embed in batches instead of one API call per chunk
            embeddings = []
            for start in range(0, len(chunks), Config.EMBED_BATCH_SIZE):
                batch = chunks[start:start + Config.EMBED_BATCH_SIZE]
                response = service_manager.cohere_client.embed(
                    texts=batch,
                    model=Config.EMBED_MODEL,
                    input_type="search_document"
                )
                embeddings.extend(response.embeddings)

            vectors_to_upsert = [
                {
                    'id': f'{safe_filename}#chunk_{i}',
                    'values': embedding,
                    'metadata': {
                        'text': chunk,
                        'filename': safe_filename,
                        'chunk_index': i
                    }
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # Upsert in batches to stay under Pinecone's request size limit
            for start in range(0, len(vectors_to_upsert), 100):
                index.upsert(
                    vectors=vectors_to_upsert[start:start + 100],
                    namespace=namespace
                )
            return len(vectors_to_upsert)

        chunks_processed = await asyncio.to_thread(embed_and_upsert)

        # Update chatbot files list (no duplicate entries on re-upload)
        if safe_filename not in chatbot_manager.chatbots[chatbot_name]["files"]:
            chatbot_manager.chatbots[chatbot_name]["files"].append(safe_filename)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"File '{safe_filename}' uploaded successfully",
                "details": {
                    "filename": safe_filename,
                    "size": file_size,
                    "chunks_processed": chunks_processed,
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

        service_manager = chatbot_manager.service_manager
        namespace = chatbot_manager.chatbots[chatbot_name]["namespace"]

        def run_rag():
            # Generate query embedding
            response = service_manager.cohere_client.embed(
                texts=[query],
                model=Config.EMBED_MODEL,
                input_type="search_query"
            )
            query_embedding = response.embeddings[0]

            # Retrieve a wide candidate set, then rerank down to the best few
            search_results = service_manager.index.query(
                vector=query_embedding,
                top_k=Config.RETRIEVE_TOP_K,
                include_metadata=True,
                namespace=namespace
            )

            matches = search_results['matches']
            if not matches:
                return None, []

            documents = [match['metadata']['text'] for match in matches]
            rerank_results = service_manager.cohere_client.rerank(
                model=Config.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(Config.RERANK_TOP_N, len(documents))
            )

            sources = [
                {
                    "text": documents[r.index],
                    "filename": matches[r.index]['metadata'].get('filename', 'unknown'),
                    "relevance": round(r.relevance_score, 3)
                }
                for r in rerank_results.results
            ]

            context = "\n\n".join(
                f"[Source: {s['filename']}]\n{s['text']}" for s in sources
            )

            if not context.strip():
                return None, []

            # Generate answer
            prompt = f"""Context:
{context}

Question: {query}

Please provide a clear and concise answer based only on the context above, mentioning which source file the answer comes from. If the context does not contain the answer, say you don't know."""

            chat_response = service_manager.cohere_client.chat(
                model=Config.CHAT_MODEL,
                message=prompt,
                max_tokens=300,
                temperature=0.3
            )
            return chat_response.text.strip(), sources

        answer, sources = await asyncio.to_thread(run_rag)

        if answer is None:
            return {"status": "error", "answer": "No relevant information found"}
        relevant_chunks = [s["text"] for s in sources]

        # Save to chat history
        chatbot_manager.save_chat_history(chatbot_name, query, answer)

        return {
            "status": "success",
            "answer": answer,
            "context": relevant_chunks,
            "sources": sources
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
