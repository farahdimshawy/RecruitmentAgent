from typing import List, Dict
from core.rag.vectorstore import retrieve_vector_data, SKILLS_INDEX_NAME

def get_matching_skills(job_description_text: str, k: int = 10, score_threshold: float = 0.65) -> List[Dict]:
    """
    Queries the skills-index with the Job Description text to retrieve a list of 
    canonical, highly relevant skill IDs.

    Args:
        job_description_text (str): The full raw text of the Job Description.
        k (int): The number of top-k results to retrieve from the vector index.
        score_threshold (float): The minimum similarity score required for a skill 
                                 to be considered relevant (0.0 to 1.0).

    Returns:
        List[Dict]: A list of dictionaries, where each dict contains the 
                    'id' (canonical skill ID), 'score', and 'content' 
                    (long-form skill definition).
    """
    if not job_description_text or len(job_description_text) < 20:
        print("Error: Job description text is too short or empty for effective skill retrieval.")
        return []
        
    print(f"--- Skill Matcher: Retrieving top {k} skills from '{SKILLS_INDEX_NAME}' ---")
    
    try:
        # 1. Query the skills index using the JD text
        results = retrieve_vector_data(
            query=job_description_text,
            k=k,
            index_name=SKILLS_INDEX_NAME
        )
        
        matches = results.get('matches', [])
        
        if not matches:
            print("No skills found that match the job description.")
            return []

        # 2. Filter matches based on the similarity score threshold
        filtered_skills = []
        for match in matches:
            score = match.get('score', 0.0)
            skill_id = match.get('id')
            skill_content = match['metadata'].get('content')
            
            if score >= score_threshold:
                filtered_skills.append({
                    'id': skill_id,
                    'score': score,
                    'content': skill_content
                })

        print(f"Found {len(filtered_skills)} skills above threshold ({score_threshold}).")
        
        return filtered_skills

    except Exception as e:
        print(f"An error occurred during skill retrieval: {e}")
        return []


if __name__ == "__main__":
    
    # Example Job Description: Focuses on Data and ML Engineering
    SAMPLE_JD = """
    Seeking a machine learning engineer proficient in deploying models 
    to production environments, leveraging Kubernetes for orchestration, 
    and optimizing Python code for data engineering pipelines.
    """
    
    matched_skills = get_matching_skills(SAMPLE_JD, k=10, score_threshold=0.65)
    
    if matched_skills:
        print("\n--- TEST RESULTS: TOP MATCHING SKILLS ---")
        for skill in matched_skills:
            # Show ID, score, and a snippet of the definition
            print(f"- {skill['id']:<20} | Score: {skill['score']:.4f}")
        print("------------------------------------------")
    else:
        print("\n--- TEST FAILED: No relevant skills found. ---")