from typing import List, Dict, Any

from langchain_core.documents import Document
from core.utils.helpers import get_embedding_model
from core.rag.vectorstore import _get_or_create_index
import os
from dotenv import load_dotenv
load_dotenv()

SKILLS_INDEX_NAME = os.environ.get("SKILLS_INDEX_NAME")
# Import vectorstore & embeddings from your module
from .vectorstore import (
    get_vectorstore,
    retrieve_vector_data,
   #retrieve_raw
)

def retrieve_context(query: str, k: int = 5) ->List[Document]:
    vs = get_vectorstore(SKILLS_INDEX_NAME)
    return vs.similarity_search(query, k = k)

def expand_skills(skills: List[str], k: int = 3) -> Dict[str, List[str]]:
    expansion = {}
    vs = get_vectorstore(SKILLS_INDEX_NAME)

    for skill in skills:
        results = vs.similarity_search(skill, k=k)

        related = []
        for doc in results:
            if doc.page_content:
                related.append(doc.page_content)
            elif "content" in doc.metadata:
                related.append(doc.metadata["content"])

        expansion[skill] = related

    return expansion


# def expand_skills(skills: List[str], k: int = 3) -> Dict[str, List[str]]:
#     """
#     retrieve related skill synonyms from vectorstore
#     """
#     expansion = {}
#     for skill in skills:
#         results = retrieve_context(skill, k=k)
#         related = [doc.page_content for doc in results]
#         expansion[skill] = related
    
#     return expansion


# def score_resume_against_job(resume_skills, job_skills, k=3):
#     scores = []
#     vs = get_vectorstore(SKILLS_INDEX_NAME) 
#     for js in job_skills:
#         for rs in resume_skills:
#             result = vs.similarity_search_with_score(js, k=1)[0]
#             _, score = result
#             scores.append(score)

#     return sum(scores) / len(scores)

def score_resume_against_job(resume_skills, job_skills, k=3):
    scores = []
    index = _get_or_create_index(SKILLS_INDEX_NAME)
    embedder = get_embedding_model()

    for js in job_skills:
        js_embedding = embedder.embed_query(js)

        res = index.query(
            vector=js_embedding,
            top_k=k,
            include_metadata=True
        )

        for match in res["matches"]:
            scores.append(match["score"])

    return sum(scores) / len(scores) if scores else 0.0

def rag_evaluate_resume(resume_skills: List[str],
                        job_description: str,
                        expansion_k: int = 3,
                        retrieval_k: int = 5) -> Dict[str, Any]:
    """
    1. extract resume skills
    2. retrieve similar skills
    3. retrieve job-related chunks
    4. compute final sim scores

     Returns:
        {
            "expanded_resume_skills": [...],
            "expanded_job_skills": [...],
            "context_docs": [...],
            "similarity_score": 0.76
        }
    """
    expanded_resume = expand_skills(resume_skills, k = expansion_k)

    job_results =  retrieve_context(job_description, k = retrieval_k)
    expanded_job_skills = [doc.page_content for doc in job_results]

    flat_resume_skills = resume_skills + sum(expanded_resume.values(),[])
    flat_job_skills = expanded_job_skills

    score = score_resume_against_job(flat_resume_skills, flat_job_skills)

    return {
        "expanded_resume_skills": flat_resume_skills,
        "expanded_job_skills": flat_job_skills,
        "context_docs": expanded_job_skills,
        "similarity_score": float(score)
    }
def debug_raw(query: str, k: int = 5):
    """
    Useful for debugging what the Pinecone index is storing.
    """
    return retrieve_vector_data(query, k=k)