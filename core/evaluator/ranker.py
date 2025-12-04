from core.rag.vectorstore import retrieve_vector_data, RECRUITMENT_DOCS_INDEX_NAME
from core.evaluator.skill_matcher import get_matching_skills
from core.utils.helpers import model, extract_name_and_summary

from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings as genai

EMBEDDING_MODEL_NAME = 'gemini-embedding-001'

def rank_local_candidates(job_description_text: str, candidate_docs: List[str], k: int = 5) -> List[Dict]:
    """
    Ranks local candidate documents against the JD using a two-stage process
    (Skill Extraction/Query and Semantic Ranking). This process is executed entirely
    in-memory without querying the persistent vector database.

    Args:
        job_description_text (str): The Job Description text.
        candidate_docs (List[str]): Raw text of local CVs/resumes.
        k (int): The number of top candidates to return.

    Returns:
        List[Dict]: A ranked list of candidate matches.
    """
    
    if not candidate_docs:
        print("[RANKER - LOCAL] No candidate documents provided.")
        return []

    print(f"\n[RANKER - LOCAL] Processing {len(candidate_docs)} local documents.")
    
    # 1. Generate the Skill-Target Query from the JD
    skill_extraction_prompt = f"""
    Analyze the following Job Description and identify the top 5 most critical technical skills, 
    key responsibilities, and required experience areas. Combine these points into a single, 
    dense query paragraph suitable for semantic search that will prioritize candidates based 
    on relevance to the JD.

    JOB DESCRIPTION:
    ---
    {job_description_text}
    ---
    """
    
    query_response = model.generate_content(
        contents=[skill_extraction_prompt],
        generation_config={"temperature": 0.1} 
    )
    skill_query_text = query_response.text.strip()
    print(f"[RANKER - LOCAL] Generated Target Query: '{skill_query_text[:80]}...'")
    
    embed_model_client = genai
    
    query_embedding = embed_model_client.embed_content(
        model=EMBEDDING_MODEL_NAME, 
        content=skill_query_text
    )['embedding']

    # Embed all documents
    document_embeddings = embed_model_client.embed_content(
        model=EMBEDDING_MODEL_NAME, 
        content=candidate_docs
    )['embedding']


    # 3. Calculate Cosine Similarity and Rank
    ranked_candidates = []
    
    import numpy as np

    q = np.array(query_embedding)                    
    docs = np.array(document_embeddings)            
    scores = docs @ q                                

    for i, (doc_text, score) in enumerate(zip(candidate_docs, scores)):
        
        candidate_id = f"local-doc-{i+1}"

        # Extract metadata (unchanged)
        name, summary = extract_name_and_summary(doc_text, doc_id=candidate_id)

        # Scale similarity to percentage and clamp between 0–100
        normalized_score = max(0, min(100, score * 100))

        ranked_candidates.append({
            'rank': 0,   # filled after sorting
            'id': candidate_id,
            'name': name,
            'match_score': normalized_score,
            'summary': summary
        })
        
    # Sort and Assign Final Ranks
    ranked_candidates.sort(key=lambda x: x['match_score'], reverse=True)
    for i, candidate in enumerate(ranked_candidates):
        candidate['rank'] = i + 1
        
    return ranked_candidates[:k]


def rank_candidates(job_description_text: str, k: int = 5, candidate_docs: Optional[List[str]] = None) -> List[Dict]:
    """
    Executes the two-stage RAG pipeline, either against the vector database
    or against locally provided candidate documents.

    Args:
        job_description_text (str): The raw text of the Job Description.
        k (int): The number of top candidates to return.
        candidate_docs (Optional[List[str]]): List of raw CV texts for local ranking mode.

    Returns:
        List[Dict]: A ranked list of candidate matches.
    """
    
    # MODE 1: LOCAL FILES RANKING
    if candidate_docs is not None:
        return rank_local_candidates(job_description_text, candidate_docs, k)
    
    # MODE 2: DATABASE RANKING (Existing Logic)
    
    # --- Stage 1: Skill Retrieval (using the function from skill_matcher.py) ---
    matching_skills = get_matching_skills(
        job_description_text=job_description_text, 
        k=15, 
        score_threshold=0.70
    )

    if not matching_skills:
        print("\n[RANKER] No relevant skills were identified from the job description. Aborting candidate search.")
        return []

    # --- Stage 2: Construct Skill-Target Query ---
    skill_contents = [skill['content'] for skill in matching_skills]
    skill_ids_list = [skill['id'] for skill in matching_skills]
    skill_query_text = " ".join(skill_contents)
    
    print(f"\n[RANKER] Identified Skills: {', '.join(skill_ids_list)}")
    print(f"[RANKER] Querying Candidate Index with {len(skill_contents)} skill definitions.")

    # --- Stage 3: Candidate Retrieval and Ranking ---
    results = retrieve_vector_data(
        query=skill_query_text,
        k=k, # Return the final top K candidates
        index_name=RECRUITMENT_DOCS_INDEX_NAME
    )

    matches = results.get('matches', [])

    if not matches:
        print("\n[RANKER] No candidates matched the skill target query.")
        return []

    # --- Stage 4: Format and Output ---
    
    ranked_candidates = []
    
    for i, match in enumerate(matches):
        candidate_id = match['id'] 
        score = match['score']
        candidate_summary = match['metadata'].get('content', 'No summary defined.')

        # Extract candidate name from the summary (simple heuristic)
        name = "Unknown"
        summary_lines = candidate_summary.split('\n')
        for line in summary_lines:
            if line.startswith("CANDIDATE: "):
                name = line.replace("CANDIDATE: ", "").strip()
                break
        
        ranked_candidates.append({
            'rank': i + 1,
            'id': candidate_id,
            'name': name,
            'match_score': score * 100, 
            'summary': candidate_summary 
        })
        
    return ranked_candidates