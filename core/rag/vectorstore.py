import os
from pinecone import Pinecone, ServerlessSpec
from typing import Dict, Any, Union
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Load all necessary environment variables first
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") 

RECRUITMENT_DOCS_INDEX_NAME = os.environ.get("DOCS_INDEX_NAME", "recruitment-docs")
SKILLS_INDEX_NAME = os.environ.get("SKILLS_INDEX_NAME", "skills-index")
DEFAULT_INDEX_NAME = RECRUITMENT_DOCS_INDEX_NAME

EMBED_DIM = 3072 # Gemini embedding dimension
EMBEDDINGS_MODEL = "models/gemini-embedding-001" 


pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Embeddings (This relies on GEMINI_API_KEY being set)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL)

print("Pinecone client and Embeddings initialized successfully.")



# --- HELPER FUNCTION: Get or Create Index ---
def _get_or_create_index(name: str) -> Union[Any, None]:
    """
    Checks if an index exists and returns it, or creates it if missing.
    """
    if pc is None:
        print("Initialization failed. Pinecone client is unavailable.")
        return None
        
    if not name:
        print("Error: Index name cannot be empty.")
        return None
    indexes = pc.list_indexes()    # returns {'indexes': [{'name': 'xxx'}]}
    existing = [i["name"] for i in indexes.get("indexes", [])]

    if name not in existing:
        print(f"Index '{name}' not found. Creating it now...")
        try:
            pc.create_index(
                name=name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{name}' created.")
        except Exception as e:
            print(f"Error creating index '{name}': {e}")
            return None
            
    return pc.Index(name)


# --- CORE FUNCTION: Document Indexing ---

def add_document(id: str, content: str, metadata: Dict[str, Any] = None, index_name: str = DEFAULT_INDEX_NAME):
    """
    Embeds content and upserts the vector to the specified index.
    """
    if embeddings is None:
        print("Embedding model is unavailable. Upsert aborted.")
        return

    index = _get_or_create_index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    if index is None:
        print(f"Failed to connect to index {index_name}. Upsert aborted.")
        return

    try:
        
        embedding = embeddings.embed_query(content)
        
        final_metadata = metadata if metadata else {}
        final_metadata["content"] = content

        index.upsert(vectors=[(id, embedding, final_metadata)])
        
    except Exception as e:
        print(f"ERROR: Failed to embed or upsert document {id} to {index_name}. Error: {e}")


# --- CORE FUNCTION: Raw Data Retrieval ---

def retrieve_vector_data(query: str, k: int = 5, index_name: str = DEFAULT_INDEX_NAME, filter: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Retrieves the top-k raw vector results from the specified index.
    """
    if embeddings is None:
        print("Embedding model is unavailable. Retrieval aborted.")
        return {"matches": []}

    index = _get_or_create_index(index_name)
    if index is None:
        return {"matches": []}
    
    try:
        # Use embed_query for a single query vector
        query_vector = embeddings.embed_query(query)
        
        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter=filter if filter else {}
        )
        return results

    except Exception as e:
        print(f"ERROR: Failed to retrieve data from {index_name}. Error: {e}")
        return {"matches": []}


# --- INDEX MANAGEMENT ---

def clear_index(name: str):
    """Delete all vectors from the specified index (use with caution)."""
    if pc is None:
        print("Initialization failed. Pinecone client is unavailable.")
        return

    index = _get_or_create_index(name)
    if index:
        try:
            index.delete(delete_all=True)
            print(f"All vectors deleted from {name}.")
        except Exception as e:
            print(f"Error clearing index {name}: {e}")

def delete_index(name: str):
    """Delete the whole Pinecone index."""
    if pc is None:
        print("Initialization failed. Pinecone client is unavailable.")
        return
        
    try:
        pc.delete_index(name)
        print(f"Index '{name}' deleted.")
    except Exception as e:
        print(f"Error deleting index {name}: {e}")