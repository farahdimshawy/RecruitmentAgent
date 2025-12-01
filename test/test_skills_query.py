import os
from core.rag.vectorstore import retrieve_vector_data, SKILLS_INDEX_NAME
from dotenv import load_dotenv

load_dotenv()

def test_skills_retrieval(query: str, k: int = 5):
    """
    Tests the retrieval process against the skills index for a given job description snippet.
    """
    print(f"--- Querying Skills Index: '{SKILLS_INDEX_NAME}' ---")
    print(f"TEST QUERY: '{query}'\n")

    # Call the core retrieval function, explicitly targeting the SKILLS_INDEX_NAME
    results = retrieve_vector_data(
        query=query, 
        k=k, 
        index_name=SKILLS_INDEX_NAME
    )

    if not results or not results.get('matches'):
        print("No matches found or an error occurred during retrieval.")
        return

    print("Top Matched Skill IDs:")
    print("----------------------")
    
    matches = results['matches']
    
    # Sort matches by score descending (they usually come pre-sorted, but good practice)
    sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)

    for match in sorted_matches:
        # The ID is the canonical skill name (e.g., CAN_NLP_LLMS)
        skill_id = match['id'] 
        score = match['score']
        
        # The metadata content is the long-form definition of the skill
        skill_definition = match['metadata'].get('content', 'No content defined.')
        
        print(f"ID: {skill_id:<30} | Score: {score:.4f}")
        print(f"   -> Definition Snippet: {skill_definition[:75]}...")
        
    print("\n--- Retrieval Test Complete ---")


if __name__ == "__main__":
    # Define a test job description snippet focused on specific technical skills
    TEST_JOB_DESCRIPTION_SNIPPET = (
        "Seeking a machine learning engineer proficient in deploying models "
        "to production environments, leveraging Kubernetes for orchestration, "
        "and optimizing Python code for data engineering pipelines."
    )
    
    test_skills_retrieval(TEST_JOB_DESCRIPTION_SNIPPET, k=8)