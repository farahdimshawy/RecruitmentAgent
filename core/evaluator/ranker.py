import os
from typing import List, Dict, Any, Tuple
from core.rag.vectorstore import retrieve_vector_data, RECRUITMENT_DOCS_INDEX_NAME
from core.evaluator.skill_matcher import get_matching_skills

def rank_candidates(job_description_text: str, k: int = 5) -> List[Dict]:
    """
    Executes the two-stage RAG pipeline:
    1. Retrieves relevant skills (Skill IDs) from the skills-index using the JD.
    2. Constructs a dense query from those skill definitions.
    3. Retrieves and ranks candidates from the recruitment-docs index using the dense query.

    Args:
        job_description_text (str): The raw text of the Job Description.
        k (int): The number of top candidates to return.

    Returns:
        List[Dict]: A ranked list of candidate matches, including their ID, 
                    score, and a summary of their profile.
    """
    
    # --- Stage 1: Skill Retrieval (using the function from skill_matcher.py) ---
    # We use a threshold of 0.70 to ensure we capture a good range of skills.
    matching_skills = get_matching_skills(
        job_description_text=job_description_text, 
        k=15, # Retrieve more skills initially
        score_threshold=0.70
    )

    if not matching_skills:
        print("\n[RANKER] No relevant skills were identified from the job description. Aborting candidate search.")
        return []

    # --- Stage 2: Construct Skill-Target Query ---
    # Combine the content (definitions) of all high-scoring skills into one dense query string.
    skill_contents = [skill['content'] for skill in matching_skills]
    skill_ids_list = [skill['id'] for skill in matching_skills]
    
    # Create a dense, focused query by concatenating the skill definitions
    skill_query_text = " ".join(skill_contents)
    
    print(f"\n[RANKER] Identified Skills: {', '.join(skill_ids_list)}")
    print(f"[RANKER] Querying Candidate Index with {len(skill_contents)} skill definitions.")

    # --- Stage 3: Candidate Retrieval and Ranking ---
    
    # Use the dense skill query text to find candidates in the recruitment-docs index
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

        # Extract Candidate Name (assuming the chunk format from document_corpus.py)
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
            'match_score': score,
            'summary': candidate_summary 
        })
        
    return ranked_candidates

# --- Example Usage for Testing ---
if __name__ == "__main__":
    
    # Example Job Description: Focuses on Data and ML Engineering
    SAMPLE_JD = """
    We are seeking a Senior Data Scientist skilled in deep learning, 
    Natural Language Processing (NLP), and deploying LLM applications. 
    The ideal candidate has strong Python engineering skills, specifically
    for creating scalable data pipelines, and experience with vector databases
    for Retrieval-Augmented Generation (RAG) systems. Must know MLOps and cloud deployment practices.
    """
    
    print("===================================================================")
    print(">>> Starting Candidate Ranking for JD: Senior Data Scientist <<<")
    print("===================================================================")
    
    top_candidates = rank_candidates(SAMPLE_JD, k=5)
    
    if top_candidates:
        print("\nRANKING SUCCESSFUL: TOP CANDIDATES")
        print("--------------------------------------------------------------------------------")
        print(f"{'Rank':<5} | {'Match Score':<12} | {'ID':<15} | {'Candidate Name'}")
        print("--------------------------------------------------------------------------------")
        
        for candidate in top_candidates:
            print(f"{candidate['rank']:<5} | {candidate['match_score']:.4f}{'<-- HIGH MATCH' if candidate['match_score'] > 0.8 else '' :<12} | {candidate['id']:<15} | {candidate['name']}")
        
        # Optionally show the summary of the top candidate
        print("\n[SUMMARY OF TOP CANDIDATE (Rank 1)]")
        print(top_candidates[0]['summary'])
        
        print("===================================================================")
    else:
        print("\nRANKING FAILED: Could not match candidates to the Job Description.")