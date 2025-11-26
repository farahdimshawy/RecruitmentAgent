# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME") 


# initialize embeddings model + vector store

EMBED_DIM = 3072 #gemini embedding dimension

def _get_or_create_index():
    # Create index if missing
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)


index = _get_or_create_index()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

def add_documents(texts: list[str]):

    docs = [Document(page_content=t) for t in texts]
    return vector_store.add_documents(docs)


def add_document(id: str, content: str):
    embedding = embeddings.embed_query(content)
    index.upsert(vectors=[(id, embedding, {"content": content})])

def retrieve(query: str, k: int = 5):
    return vector_store.similarity_search(query, k=k)

def retrieve_raw(query: str, k: int = 5):
    """
    Retrieval without LangChain, returns raw Pinecone results.
    """
    vector = embeddings.embed_query(query)
    return index.query(vector=vector, top_k=k, include_metadata=True)

def retrieve_context(query: str, k: int = 5):
    """Retrieve top-k relevant documents from vectorstore."""
    return vector_store.similarity_search(query, k=k)

def load_texts_from_folder(folder):
    texts = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".txt"):
            with open(path, "r") as f:
                texts.append(f.read())
    return texts

def clear_index():
    """Delete all vectors."""
    index.delete(deleteAll='true')


def delete_index():
    """Delete the whole Pinecone index."""
    pc.delete_index(index_name)

def get_vectorstore():
    return vector_store
