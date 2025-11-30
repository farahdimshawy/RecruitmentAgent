import pandas as pd
import os
import time
from typing import Dict, Any

# --- Imports from your environment ---
# NOTE: Ensure these imports correctly point to your modules
from core.utils.helpers import model 
from core.utils.to_native import to_native
from core.rag.vectorstore import add_document 

from google.generativeai.types import FunctionDeclaration, Tool


# --- YOUR CV PARSER (Integrated) ---
def gem_json(text: str) -> Dict[str, Any]:
    """
    Parses a CV string using Gemini function calling to extract structured data.
    """
    # Create the prompt for the model
    extraction_prompt = f"""
    Please analyze the following CV and extract the required information.
    Here is the CV:
    ---
    {text}
    ---
    """

    # Define the Function Declaration with your rich schema
    extract_cv_details_func = FunctionDeclaration(
        name="extract_cv_details",
        description="Extracts key details from a CV text.",
        parameters = {
            "type": "object",
            "properties": {
                "Name": {"type": "string", "description": "The applicant's full name"},
                "Contact_Info": {
                    "type": "object",
                    "properties": {
                        "Email": {"type": "string"},
                        "Phone": {"type": "string"},
                        "LinkedIn": {"type": "string"},
                        "Portfolio": {"type": "string"}
                    }
                },
                "Education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Degree": {"type": "string"},
                            "Major": {"type": "string"},
                            "Institution": {"type": "string"},
                            "Graduation_Year": {"type": "string"},
                            "GPA": {"type": "string"}
                        }
                    }
                },
                "Experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Title": {"type": "string"},
                            "Company": {"type": "string"},
                            "Duration": {"type": "string"},
                            "Responsibilities": {"type": "string"},
                            "Technologies": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "Projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Title": {"type": "string"},
                            "Description": {"type": "string"},
                            "Technologies": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "Skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Technical and soft skills (e.g., Python, Machine Learning, Communication)"
                },
                "Certifications": {"type": "array", "items": {"type": "string"}},
                "Languages": {"type": "array", "items": {"type": "string"}},
                "Career_Objective": {
                    "type": "string",
                    "description": "Short statement about the applicant's professional goals"
                },
                "Soft_Skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Non-technical skills such as leadership, teamwork, or communication"
                },
                "Location": {
                    "type": "string",
                    "description": "Applicant's current city or country"
                },
                "Availability": {
                    "type": "string",
                    "description": "Whether the applicant is available full-time, part-time, or for internships"
                }
            },
            "required": ["Name", "Education", "Skills"]
        }
    )
    
    review_tool = Tool(function_declarations=[extract_cv_details_func])

    # Implementing exponential backoff for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                extraction_prompt,
                tools=[review_tool],
                tool_config={'function_calling_config': 'ANY'}
            )
            function_call_part = response.candidates[0].content.parts[0]
            function_call = function_call_part.function_call

            function_args = dict(function_call.args)
            native_data = to_native(function_args)
            
            return native_data
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # print(f"API call failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"LLM Extraction Error after {max_retries} attempts: {e}")
                return None


# --- PIPELINE BUILDER ---
DOCUMENT_INDEX_NAME = os.environ.get("DOCS_INDEX_NAME") 

def build_document_corpus(csv_filepath: str, resume_col: str, id_col: str, limit: int = 20):
    """
    Reads CSV, extracts rich data using gem_json, formats into a text chunk 
    (Augmented Chunking), and embeds into the Document Corpus index.
    """
    
    if not os.path.exists(csv_filepath):
        print(f"File not found: {csv_filepath}")
        return

    print(f"Reading {csv_filepath}...")
    df = pd.read_csv(csv_filepath).head(limit)
    print(f"Processing {len(df)} resumes and indexing into '{DOCUMENT_INDEX_NAME}'...")

    for index, row in df.iterrows():
        raw_text = str(row.get(resume_col, ""))
        # Use a consistent candidate ID based on the CSV column
        cand_id = str(row.get(id_col, f"unknown_id_{index}"))
        
        if len(raw_text) < 10: 
            print(f"Skipping row {index}: Text too short.")
            continue

        # 1. EXTRACT STRUCTURED DATA using your LLM function
        data = gem_json(raw_text)
        
        if not data:
            print(f"Skipping row {index}: Extraction failed.")
            continue

        # 2. CREATE AUGMENTED CHUNK (High-Signal Text for Embedding)
        
        # Aggregate Experience Details
        exp_list = []
        for job in data.get('Experience', []):
            title = job.get('Title', 'N/A')
            comp = job.get('Company', '')
            tech = ", ".join(job.get('Technologies', []))
            exp_list.append(f"{title} at {comp} [{tech}]")
        exp_str = "; ".join(exp_list)

        # Aggregate Skills and Education
        skills_str = ", ".join(data.get('Skills', []))
        edu_list = [f"{e.get('Degree', '')} in {e.get('Major', '')}" for e in data.get('Education', [])]
        edu_str = "; ".join(edu_list)

        # Construct the final dense chunk for the vector model to embed
        vector_content = f"""
        CANDIDATE: {data.get('Name', 'Unknown')}
        LOCATION: {data.get('Location', 'Unknown')}
        OBJECTIVE: {data.get('Career_Objective', '')}
        TOP SKILLS: {skills_str}
        EXPERIENCE SUMMARY: {exp_str}
        EDUCATION SUMMARY: {edu_str}
        """

        # 3. EMBED & STORE
        try:
            add_document(id=cand_id, content=vector_content.strip(), index_name= DOCUMENT_INDEX_NAME)
            print(f"[{index+1}/{len(df)}] Indexed: {cand_id}")
            
            # Simple delay to respect API rate limits
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error indexing {cand_id}: {e}")

    print("\n--- Document Corpus Build Complete ---")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Ensure you have a 'rag_corpus_resumes.csv' file available
    CSV_FILE = "./data/rag_corpus/rag_corpus.csv" 
    # Change these strings to match the column names in your CSV file
    RESUME_TEXT_COLUMN = "text"
    ID_COLUMN = "id"

    build_document_corpus(CSV_FILE, RESUME_TEXT_COLUMN, ID_COLUMN)

