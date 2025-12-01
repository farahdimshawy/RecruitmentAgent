import pandas as pd
import os
import time
from typing import Dict, Any

from core.utils.helpers import model 
from core.utils.to_native import to_native
from core.rag.vectorstore import add_document 
from core.extractor.cv_parser import cv_parser

from google.generativeai.types import FunctionDeclaration, Tool



DOCUMENT_INDEX_NAME = os.environ.get("DOCS_INDEX_NAME") 

def build_document_corpus(csv_filepath: str, resume_col: str, id_col: str, limit: int = 100):
    """
    Reads CSV, extracts rich data using cv_parser, formats into a text chunk 
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
        data = cv_parser(raw_text)
        
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
    
    CSV_FILE = "./data/rag_corpus/tech_corpus.csv" 
    # Change these strings to match the column names in your CSV file
    RESUME_TEXT_COLUMN = "text"
    ID_COLUMN = "id"

    build_document_corpus(CSV_FILE, RESUME_TEXT_COLUMN, ID_COLUMN)

