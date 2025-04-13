# rag_fastapi.py
import os
import time
from typing import List, Dict, Any

import fitz  # PyMuPDF for PDF processing
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# LangChain components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Pinecone components
from pinecone import Pinecone, ServerlessSpec, Index  # Ensure correct import

# Embedding components
from sentence_transformers import SentenceTransformer

# sklearn components (for placeholder functions)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------------------------------
# Configuration Variables
# ---------------------------------------
PREDEFINED_PDF_PATH = "data/Sidharth Khandelwal_M_34_2024_12.pdf"
PINECONE_API_KEY = "pcsk_dvpoQ_dm3VnkQ7wY2Kdp1mkhP6WorxFkwkCe4Wxb1JkCNaAdzBweuw5shrZCai5fNa4B"
GOOGLE_GEMINI_API_KEY = "AIzaSyA7zT1-4oEyPcToeKETl4-wPK07EvQ5sUM"

INDEX_NAME = "clinical-hybrid-rag"  # Pinecone index name
DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # dense model
DIMENSION = 768  # embedding dimension
BATCH_SIZE = 32  # for embedding generation and upsert
TOP_K_RETRIEVAL = 15  # number of results to fetch
CHUNK_SIZE = 1000  # text chunk size (for PDF splitting)
CHUNK_OVERLAP = 150  # overlap of chunks

# ---------------------------------------
# Global Components (to be initialized at startup)
# ---------------------------------------
pc = None  # Pinecone client
embedder = None  # SentenceTransformer
memory = None  # ConversationBufferMemory
rag_chain = None  # Conversational RAG chain
retriever = None  # PineconeRetriever instance
llm = None  # Google Gemini LLM instance

# ---------------------------------------
# Helper / Placeholder Functions
# ---------------------------------------
def generate_sparse_vectors(text):
    """Generates sparse vectors using TF-IDF vectorization. (Not used)"""
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    sparse_matrix = tfidf_vectorizer.fit_transform([text])
    coo_matrix = sparse_matrix.tocoo()
    sparse_vector = {
        "indices": coo_matrix.col.tolist(),
        "values": coo_matrix.data.tolist()
    }
    return sparse_vector

def generate_sparse_query_vector(query: str) -> Dict[str, Any]:
    """Placeholder for generating a sparse query vector (Not used)."""
    print("\n--->>> WARNING: Using PLACEHOLDER sparse query vector generation! <<<---")
    indices = list(range(min(len(query), 20)))
    values = [round(0.1 * (j + 1), 4) for j in range(len(indices))]
    if not indices:
        indices = [0]
        values = [0.0]
    return {"indices": indices, "values": values}

# ---------------------------------------
# PDF Loading and Chunking
# ---------------------------------------
def load_and_chunk_pdf(file_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    print(f"Starting PDF processing for: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    doc = fitz.open(file_path)
    source_metadata = {"source": os.path.basename(file_path)}
    documents = []
    print(f"Reading {len(doc)} pages...")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            page_meta = source_metadata.copy()
            page_meta["page"] = page_num + 1
            documents.append(Document(page_content=text, metadata=page_meta))
    doc.close()

    if not documents:
        raise ValueError("No text extracted from the PDF.")
    print(f"Extracted text from {len(documents)} pages.")

    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    split_docs = splitter.split_documents(documents)
    chunks = []
    min_chunk_len = 50
    for i, chunk_doc in enumerate(split_docs):
        cleaned_text = chunk_doc.page_content.strip()
        if len(cleaned_text) >= min_chunk_len:
            chunks.append({
                "id": f"{source_metadata['source']}-chunk-{i}",
                "text": cleaned_text,
                "metadata": {
                    "source": chunk_doc.metadata.get("source"),
                    "page": chunk_doc.metadata.get("page")
                }
            })
    print(f"Chunked PDF into {len(chunks)} valid chunks (min length {min_chunk_len}).")
    return chunks

# ---------------------------------------
# Dense Embedding Generation
# ---------------------------------------
def generate_dense_embeddings(texts: List[str]) -> List[List[float]] | None:
    print(f"Generating dense embeddings for {len(texts)} text chunks...")
    try:
        embeddings = embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
        print("Dense embedding generation complete.")
        return embeddings
    except Exception as e:
        print(f"ERROR: Dense embedding generation failed: {e}")
        return None

# ---------------------------------------
# Upsert Vectors into Pinecone
# ---------------------------------------
def upsert_to_pinecone(index: Index, chunks: List[Dict[str, Any]]):
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        # Generate dense embedding for each chunk
        dense_vector = embedder.encode([chunk["text"]], convert_to_numpy=True)[0].tolist()
        metadata = {
            "text": chunk["text"],
            "page": chunk["metadata"].get("page", 0),
            "source": chunk["metadata"].get("source", "unknown")
        }
        vectors_to_upsert.append({
            "id": f"doc_{i}",
            "values": dense_vector,
            "metadata": metadata
        })
    print(f"Attempting to upsert {len(vectors_to_upsert)} vectors...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
    print("Vector upsert complete.")

# ---------------------------------------
# Pinecone Index Initialization
# ---------------------------------------
def initialize_pinecone_index() -> Index:
    global pc
    try:
        print("Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("Pinecone connection successful.")
    except Exception as e:
        raise Exception(f"Error initializing Pinecone: {e}")
    try:
        print(f"Checking Pinecone index '{INDEX_NAME}'...")
        index_list_response = pc.list_indexes()
        existing_index_names = [index_info['name'] for index_info in index_list_response.indexes]
        print(f"Found existing indexes: {existing_index_names}")
        if INDEX_NAME not in existing_index_names:
            print(f"Creating new index '{INDEX_NAME}' with dimension {DIMENSION} and cosine metric...")
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=spec
            )
            print(f"Index '{INDEX_NAME}' creation initiated. Waiting for it to become available...")
            time.sleep(60)
        else:
            print(f"Found existing index '{INDEX_NAME}'.")
        print(f"Connecting to index '{INDEX_NAME}'...")
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"Connected to index '{INDEX_NAME}'. Stats: {stats}")
        return index
    except Exception as e:
        raise Exception(f"Failed to initialize or connect to Pinecone index '{INDEX_NAME}': {e}")

# ---------------------------------------
# Pinecone Retriever Class
# ---------------------------------------
class PineconeRetriever:
    def __init__(self, index: Index, dense_embedder: SentenceTransformer):
        self.index = index
        self.dense_embedder = dense_embedder

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
        print(f"\nPerforming dense search for query: '{query}' (top_k={top_k})")
        try:
            print("Generating dense query vector...")
            dense_query_vector = self.dense_embedder.encode(query).tolist()
            if not dense_query_vector:
                print("Warning: Could not generate dense query vector.")
                return "[Error: Failed to generate dense query vector]"
            print("Executing Pinecone query (dense search only)...")
            results = self.index.query(
                vector=dense_query_vector,
                top_k=top_k,
                include_metadata=True
            )
            matches = results.get('matches', [])
            print(f"Retrieved {len(matches)} results from Pinecone.")
            context_list = [
                match.metadata.get("text", "")
                for match in matches if match.metadata.get("text")
            ]
            if not context_list:
                print("Warning: No text found in metadata of retrieved results.")
                return "[No relevant context found in the report for this query]"
            context = "\n\n---\n\n".join(context_list)
            return context
        except Exception as e:
            print(f"ERROR during retrieval from Pinecone: {e}")
            return "[Error during context retrieval]"

# ---------------------------------------
# Conversational RAG Chain Setup
# ---------------------------------------
def create_conversational_rag_chain(
    llm: ChatGoogleGenerativeAI,
    retriever: PineconeRetriever,
    memory: ConversationBufferMemory
):
    print("Setting up conversational RAG chain...")
    template = """You are a helpful AI assistant collaborating with a medical professional.
You have access to relevant sections of a patient's medical report based on the current conversation topic.
Answer the doctor's questions concisely and clearly, referencing the report context when necessary.
Maintain a conversational and collaborative tone. If the context is missing information to answer, state that clearly.

**Conversation History:**
{chat_history}

**Relevant Information from Report:**
{context}

**Doctor:** {input}

**Assistant:**"""
    prompt = ChatPromptTemplate.from_template(template)

    def retrieve_and_print_context(input_dict: Dict) -> str:
        return retriever.retrieve(query=input_dict["input"])

    conversational_rag_chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | (lambda x: x['chat_history'])
        )
        | RunnablePassthrough.assign(
            context=retrieve_and_print_context
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Conversational RAG chain setup complete.")
    return conversational_rag_chain

# ---------------------------------------
# FastAPI Application Setup
# ---------------------------------------
app = FastAPI(title="Clinical Report Assistant RAG")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

@app.on_event("startup")
def startup_event():
    global embedder, memory, rag_chain, retriever, llm, pc

    # Initialize the Pinecone connection and dense embedder
    try:
        print("Initializing Pinecone and dense embedding model...")
        index = initialize_pinecone_index()
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        embedder = SentenceTransformer(DENSE_MODEL_NAME, device=device)
        print(f"Dense embedder loaded successfully on device '{device}'.")
    except Exception as e:
        raise Exception(f"Startup error during component initialization: {e}")

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print("Conversation memory initialized.")

    # Process the PDF and upsert embeddings if index is empty
    try:
        index_stats = index.describe_index_stats()
        if index_stats.total_vector_count == 0:
            print("Index is empty. Processing and uploading PDF data...")
            chunks = load_and_chunk_pdf(PREDEFINED_PDF_PATH)
            if not chunks:
                raise Exception("No chunks generated from PDF. Cannot proceed.")
            upsert_to_pinecone(index, chunks)
            print("Report processing and indexing complete.")
            index_stats = index.describe_index_stats()
            print(f"Index now contains {index_stats.total_vector_count} vectors.")
        else:
            print(f"Index already contains {index_stats.total_vector_count} vectors. Skipping PDF processing.")
    except Exception as e:
        raise Exception(f"Error during PDF processing or upsert: {e}")

    # Initialize Pinecone retriever
    retriever = PineconeRetriever(index, embedder)
    print("Pinecone retriever initialized for dense search.")

    # Initialize Google Gemini LLM
    try:
        print("Initializing Google Gemini LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.4,
            max_tokens=2000,
            api_key=GOOGLE_GEMINI_API_KEY
        )
        print("Google Gemini LLM initialized.")
    except Exception as e:
        raise Exception(f"Failed to initialize Google Gemini LLM: {e}")

    # Create the conversational RAG chain
    rag_chain = create_conversational_rag_chain(llm, retriever, memory)

@app.post("/ask", response_model=QueryResponse)
def ask_query(query_request: QueryRequest):
    global rag_chain, memory
    question = query_request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        print(f"\nReceived query: {question}")
        inputs = {"input": question}
        response = rag_chain.invoke(inputs)
        memory.save_context(inputs, {"output": response})
        print(f"Responding with: {response}")
        return QueryResponse(response=response)
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

@app.get("/")
def read_root():
    return {"message": "Clinical Report Assistant RAG is up and running. Post a JSON with key 'question' to /ask."}

# ---------------------------------------
# Run the FastAPI app via: uvicorn rag_fastapi:app --reload
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_fastapi:app", host="0.0.0.0", port=8000, reload=True)
