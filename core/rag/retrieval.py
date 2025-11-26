from core.utils.helpers import model
from core.utils.to_native import to_native
import os
from typing import List, Dict, Any

from langchain_core.documents import Document

# Import vectorstore & embeddings from your module
from .vectorstore import (
    get_vectorstore,
    embeddings,
    retrieve,
    retrieve_raw
)

def retrieve_context(query: str, k: int = 5) ->List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query, k = k)

def expand_skills(skills: List[str], k: int = 3) -> Dict[str, List[str]]:
    """
    retrieve related skill synonyms from vectorstore
    """
    expansion = {}
    for skill in skills:
        results = retrieve_context(skill, k=k)
        related = [doc.page_content for doc in results]
        expansion[skill] = related
    
    return expansion

# def compute_similarity(skill: str, target: str) -> float:

#     v1 = embeddings.embed_query(skill)
#     v2 = embeddings.embed_query(target)

#     dot = sum(a*b for a,b in zip(v1,v2))
#     norm1 = sum(a*a for a in v1) ** 0.5
#     norm2 = sum(a*a for a in v2) ** 0.5

#     return dot / (norm1 * norm2 + 1e-9)

def score_resume_against_job(resume_skills, job_skills, k=3):
    scores = []
    vs = get_vectorstore() 
    for js in job_skills:
        for rs in resume_skills:
            result = vs.similarity_search_with_score(js, k=1)[0]
            _, score = result
            scores.append(score)

    return sum(scores) / len(scores)


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
    return retrieve_raw(query, k=k)