import os
import time
from typing import Dict, Any
from core.evaluator.ranker import rank_candidates # Import the ranker to get data
from core.utils.helpers import model 
from google.api_core.exceptions import GoogleAPIError

from dotenv import load_dotenv

load_dotenv()

def generate_outreach_email(job_description: str, candidate_data: Dict[str, Any], recruiter_name: str = "Recruitment Agent") -> str:
    """
    Uses the Gemini model to generate a personalized recruitment outreach email
    based on the candidate's profile summary and the job description.

    Args:
        job_description (str): The raw text of the job description.
        candidate_data (Dict[str, Any]): A single candidate's ranked data 
                                         (from rank_candidates output).
        recruiter_name (str): The name to sign the email with.

    Returns:
        str: The generated email content, ready to be copied.
    """
    
    candidate_name = candidate_data.get('name', 'Talented Professional')
    candidate_summary = candidate_data.get('summary', 'Profile summary not available.')
    match_score = candidate_data.get('match_score', 0.0)

    # 1. Define the System Instruction (Persona and Formatting)
    system_prompt = (
        "You are a highly professional, friendly, and enthusiastic technical recruiter. "
        "Your goal is to write a personalized outreach email to a candidate based on "
        "their CV summary and a target job description. "
        "The email must be concise, professional, and directly state why the candidate is a strong fit, "
        "using the skill keywords found in the summary. Do not include a subject line."
    )

    # 2. Define the User Query (The Content)
    user_prompt = f"""
    Generate an outreach email draft using the following information:

    1. **Candidate Name:** {candidate_name}
    2. **Candidate Profile Summary (Augmented Data):**
       ---
       {candidate_summary}
       ---
    3. **Target Job Description:**
       ---
       {job_description}
       ---
    4. **Recruiter Name:** {recruiter_name}
    5. **Match Score:** {match_score:.2f} (Use this to gauge enthusiasm, but do not state the score directly in the email.)

    Start the email with a greeting and end with a call to action (a quick chat).
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            email_agent = model.start_chat(history=[
            {"role": "system", "parts": [system_prompt]}
        ])

            response = model.generate_content(
                contents=[user_prompt],
                # Set temperature low for professional, non-creative tone
                generation_config={"temperature": 0.3} 
            )

            # Clean up the output slightly
            return response.text.strip()
            
        except GoogleAPIError as e:
            wait_time = 2 ** attempt
            print(f"API Error (Attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"General Error during email generation: {e}")
            break
            
    return "Error: Could not generate email after multiple retries."


# --- Example Usage for Testing ---
if __name__ == "__main__":
    
    # --- MOCK DATA SETUP ---
    # We must call rank_candidates first to get real data.
    SAMPLE_JD = """
    We are seeking a Senior Data Scientist skilled in deep learning, 
    Natural Language Processing (NLP), and deploying LLM applications. 
    The ideal candidate has strong Python engineering skills, specifically
    for creating scalable data pipelines, and experience with vector databases
    for Retrieval-Augmented Generation (RAG) systems. Must know MLOps and cloud deployment practices.
    """
    
    print("==========================================================")
    print(">>> Stage 1: Running Ranker to Find Top Candidate... <<<")
    print("==========================================================")
    
    top_candidates = rank_candidates(SAMPLE_JD, k=1)
    
    if not top_candidates:
        print("\nCANNOT GENERATE EMAIL: Ranker failed to find a candidate.")
    else:
        top_candidate = top_candidates[0]
        
        print(f"\nTop Candidate Found: {top_candidate['name']} (ID: {top_candidate['id']})")
        print("==========================================================")
        print(">>> Stage 2: Generating Personalized Outreach Email... <<<")
        print("==========================================================")
        
        # Generate the email for the top candidate
        email_content = generate_outreach_email(
            job_description=SAMPLE_JD,
            candidate_data=top_candidate,
            recruiter_name="Aura (Recruitment Agent)"
        )

        print("\n--- GENERATED EMAIL DRAFT ---")
        print("-----------------------------------------------------------------")
        print(f"[Subject: Exciting Senior Data Scientist Opportunity at Our Company]")
        print(email_content)
        print("-----------------------------------------------------------------")