import google.generativeai as genai
import os
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# API Key is retrieved from the environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # NOTE: The Canvas environment usually handles the API key, but we ensure it's configured.
    print("Warning: GOOGLE_API_KEY not found. Ensure environment is configured.")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the main model for content generation
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Helper Functions ---

def get_embedding_client():
    """
    Returns the configured Gemini Embedding Model client instance.
    Aligned with the user's vector store configuration ('gemini-embedding-001').
    """
    return genai

def extract_name_and_summary(doc_text: str, doc_id: str) -> Tuple[str, str]:
    """
    Uses the Gemini model to extract the candidate's name and generate a concise 
    professional summary from the raw CV text.

    Args:
        doc_text (str): The raw text content of the CV.
        doc_id (str): The unique ID for the document (used as fallback).

    Returns:
        Tuple[str, str]: (Candidate Name, Summary)
    """
    
    extraction_prompt = f"""
    Analyze the following raw Candidate Resume text.
    
    1. Identify the full professional name of the candidate.
    2. Generate a concise, professional summary (max 3 sentences) that highlights their primary job role, years of experience (if mentioned), and key technical expertise.

    Return only the name and the summary text, separated by a unique delimiter: '|||'.

    RAW RESUME TEXT:
    ---
    {doc_text}
    ---
    """

    try:
        response = model.generate_content(
            contents=[extraction_prompt],
            generation_config={"temperature": 0.1}
        )
        
        # Split the response by the delimiter
        parts = response.text.strip().split('|||')
        
        if len(parts) >= 2:
            name = parts[0].strip()
            summary = parts[1].strip()
        else:
            # Fallback if delimiter parsing fails
            name = doc_text.split('\n')[0].strip() if doc_text else f"Candidate {doc_id}"
            summary = "LLM extraction failed. Please review the full document."

        return name, summary
        
    except Exception as e:
        print(f"Error during LLM extraction for document {doc_id}: {e}")
        # Robust fallback
        name = doc_text.split('\n')[0].strip() if doc_text else f"Candidate {doc_id}"
        summary = "LLM extraction failed. Using heuristic fallback."
        return name, summary

# You may add other helper functions (e.g., chunking, logging) here later.