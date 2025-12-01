import streamlit as st
import pandas as pd
import sys
import os
from typing import List, Dict, Optional


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

st.set_page_config(
    page_title="RAG Recruitment Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

from core.extractor.pdf_reader import pdf_extract_text_from_bytes
from core.evaluator.ranker import rank_candidates
from core.evaluator.email_generator import generate_outreach_email


def display_ranked_candidates(ranked_candidates: List[Dict]):
    """Displays the ranked candidate data in a clean, interactive table."""
    if not ranked_candidates:
        st.warning("No candidates were found matching the job requirements.")
        return

    st.header("🏆 Top Matched Candidates")
    st.markdown("Results are ranked by the calculated vector similarity score against the extracted skills from the JD.")

    data = []
    
    for candidate in ranked_candidates:
        data.append({
            'Rank': candidate['rank'],
            'Match Score (%)': f"{candidate['match_score']:.2f}",
            'Candidate Name': candidate['name'],
            'Candidate ID': candidate['id'],
            'Summary Snippet': candidate['summary'].split('\n')[0] + '...',
            'Full Summary': candidate['summary']
        })

    df = pd.DataFrame(data)
    
    # Custom styling for the score column
    def color_score(val):
        # Handle the percentage string
        score = float(val.replace('%', ''))
        if score >= 85:
            color = 'background-color: #d1e7dd; color: #0f5132' # Green
        elif score >= 70:
            color = 'background-color: #fff3cd; color: #664d03' # Yellow
        else:
            color = 'background-color: #f8d7da; color: #842029' # Red
        return color
    
    styled_df = df[['Rank', 'Match Score (%)', 'Candidate Name', 'Summary Snippet']].style.map(
        color_score, subset=['Match Score (%)']
    ).set_properties(
        subset=['Summary Snippet'], **{'white-space': 'normal', 'text-align': 'left'}
    )

    st.dataframe(styled_df, width='stretch', hide_index=True)

    candidate_names = [cand['Candidate Name'] for cand in data]
    selected_name = st.selectbox(
        "Select a candidate to view detailed summary and outreach email:", 
        options=candidate_names
    )
    
    if selected_name:

        selected_candidate = next(cand for cand in data if cand['Candidate Name'] == selected_name)
        st.subheader(f"Detailed Profile for {selected_name}")
        st.code(selected_candidate['Full Summary'], language='markdown')
        
        # --- Email Generation Action ---
        st.subheader("✉️ Generate Outreach Email")
        if st.button(f"Generate Email for {selected_name}", key="generate_email_btn"):
            st.session_state.email_candidate = selected_candidate
            
            with st.spinner(f"Generating personalized email for {selected_name}..."):
               
                job_description_text = st.session_state.job_description_text 
                full_candidate_data = next(
                    item for item in ranked_candidates if item['name'] == selected_name
                )
                
                email_draft = generate_outreach_email(
                    job_description=job_description_text,
                    candidate_data=full_candidate_data, 
                    recruiter_name="Your Name Here"
                )
                
                st.session_state.email_draft = email_draft
                st.session_state.email_recipient = selected_name

        if 'email_draft' in st.session_state and st.session_state.email_recipient == selected_name:
            st.success("Email Draft Generated!")
            st.code(f"[Subject: Exciting Career Opportunity for {selected_name}]", language='markdown')
            st.markdown(st.session_state.email_draft)
            st.download_button(
                label="Download Email Draft (TXT)",
                data=st.session_state.email_draft,
                file_name=f"outreach_{selected_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )


def run_ranking(job_description_text: str, top_k: int, candidate_docs: Optional[List[str]] = None):
    """
    Handles the core RAG scoring process.
    If candidate_docs is provided, it ranks against those documents.
    Otherwise, it assumes the core logic queries the vector index.
    """
    st.session_state.ranked_candidates = []
    
    # Decide spinner text based on mode
    mode = "local documents" if candidate_docs is not None else "vector index"
    spinner_text = f"Executing Two-Stage RAG Pipeline using {mode}..."
    
    # Store the documents list for display/debugging if needed
    if candidate_docs is not None:
        st.info(f"Processing {len(candidate_docs)} candidate documents locally.")
    
    with st.spinner(spinner_text):
        # Pass the candidate_docs list to the ranking function
        ranked_candidates_list = rank_candidates(
            job_description_text, 
            k=top_k, 
            candidate_docs=candidate_docs # NEW ARGUMENT
        )
        
        if ranked_candidates_list:
            st.session_state.ranked_candidates = ranked_candidates_list
            st.success("Ranking Complete! Results are below.")
        else:
            st.error("Ranking failed or no suitable candidates were found.")


# --- MAIN APP LAYOUT ---
def main():
    st.title("RAG Recruitment Agent")
    st.markdown("Upload a Job Description to find and rank the most suitable candidates from your existing vector index or locally uploaded CVs.")

    # Sidebar for Configuration
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Number of Candidates to Return (K)", min_value=1, max_value=20, value=5)
        st.markdown("**Note:** If using the database, CV data is assumed to be already indexed in the `recruitment-docs` index.")

    # --- NEW: Data Source Selection ---
    st.subheader("0. Select Candidate Data Source")
    data_source_mode = st.radio(
        "Choose where to find candidates:",
        ["Use Current Database (Vector Index)", "Upload Local CVs/Resumes"],
        key="data_source_mode",
        horizontal=True
    )
    st.markdown("---")


    candidate_texts = None
    
    # --- NEW: CV Upload Section ---
    if data_source_mode == "Upload Local CVs/Resumes":
        st.subheader("A. Upload Candidate CVs")
        cv_files = st.file_uploader(
            "Upload multiple CVs (.pdf, .txt, .md)", 
            type=['txt', 'md', 'pdf'], 
            accept_multiple_files=True,
            key="cv_uploader"
        )
        
        if cv_files:
            candidate_texts = []
            file_count = 0
            
            with st.spinner(f"Reading {len(cv_files)} CV files..."):
                for uploaded_file in cv_files:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    try:
                        if file_extension in ['txt', 'md']:
                            text = uploaded_file.read().decode("utf-8")
                            candidate_texts.append(text)
                            file_count += 1
                        
                        elif file_extension == 'pdf':
                            pdf_bytes = uploaded_file.read()
                            text = pdf_extract_text_from_bytes(pdf_bytes)
                            candidate_texts.append(text)
                            file_count += 1
                    
                    except ImportError:
                        st.error(f"Failed to process {uploaded_file.name}. PDF support requires PyMuPDF.")
                        continue
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        continue
            
            if candidate_texts:
                st.success(f"Successfully loaded text from {file_count} documents.")
            else:
                st.warning("No candidate text was successfully loaded.")
        st.markdown("---")

    # 1. Job Description Input (Key Input)
    st.subheader("1. Input Job Description (JD)")
    jd_file = st.file_uploader("Upload JD (.txt, .md or .pdf)", type=['txt', 'md', 'pdf'])
    
    job_description_text = ""
    if jd_file is not None:
        file_extension = jd_file.name.split('.')[-1].lower()
        
        with st.spinner(f"Reading {file_extension.upper()} file..."):
            if file_extension in ['txt', 'md']:
                try:
                    job_description_text = jd_file.read().decode("utf-8")
                    st.info(f"{file_extension.upper()} file uploaded successfully. Ready to rank.")
                except Exception:
                    st.error(f"Could not read {file_extension.upper()} file content.")
            
            elif file_extension == 'pdf':
                try:
                    # Read the file content as bytes
                    pdf_bytes = jd_file.read()
                    # Use the integrated PDF extraction function
                    job_description_text = pdf_extract_text_from_bytes(pdf_bytes)
                    st.info("PDF uploaded and text extracted successfully. Ready to rank.")
                except ImportError:
                    st.error("PDF support requires PyMuPDF (`fitz`) to be installed. Cannot process JD.")
                except Exception as e:
                    st.error(f"Error processing JD PDF: {e}")
                    job_description_text = ""
        
        st.session_state.job_description_text = job_description_text
    
    # Fallback to hardcoded JD for easy testing if no file is uploaded
    if not job_description_text:
        # Define a test job description for ease of use
        TEST_JD = """
        We are seeking a Senior Data Scientist skilled in deep learning, 
        Natural Language Processing (NLP), and deploying LLM applications. 
        The ideal candidate has strong Python engineering skills, specifically
        for creating scalable data pipelines, and experience with vector databases
        for Retrieval-Augmented Generation (RAG) systems. Must know MLOps and cloud deployment practices.
        """
        job_description_text = st.text_area(
            "Or paste the JD text here (uses a default JD if empty):", 
            value=TEST_JD, height=300
        )
        st.session_state.job_description_text = job_description_text


    # 2. Ranking Action
    st.subheader("2. Run Candidate Ranking")

    # Determine button disabled state
    is_database_mode = (data_source_mode == "Use Current Database (Vector Index)")
    is_upload_mode_ready = (data_source_mode == "Upload Local CVs/Resumes" and candidate_texts)
    
    is_ready_to_run = (job_description_text and (is_database_mode or is_upload_mode_ready))

    if st.button("Run RAG Scoring Pipeline", type="primary", disabled=not is_ready_to_run):
        run_ranking(job_description_text, top_k, candidate_docs=candidate_texts)

    # 3. Display Results
    if 'ranked_candidates' in st.session_state and st.session_state.ranked_candidates:
        display_ranked_candidates(st.session_state.ranked_candidates)

if __name__ == '__main__':
    main()