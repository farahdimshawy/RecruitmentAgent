import streamlit as st
import pandas as pd
import time
import sys
import os
from typing import List, Dict, Any


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

st.set_page_config(
    page_title="RAG Recruitment Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from core.evaluator.ranker import rank_candidates
    from core.evaluator.email_generator import generate_outreach_email
except ImportError as e:
    st.warning(f"Core logic not found. Proceeding with dummy functions. Check core/rag and core/communication paths.")
    print(f"Import Error details: {e}")
    rank_candidates = lambda jd, k: []  # Fallback dummy function
    generate_outreach_email = lambda jd, cand, name: "Error: Generation function not found."


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
    
    # styling for the score column
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
    
    # NOTE: Replacing applymap with map to resolve the FutureWarning.
    styled_df = df[['Rank', 'Match Score (%)', 'Candidate Name', 'Summary Snippet']].style.map(
        color_score, subset=['Match Score (%)']
    ).set_properties(
        subset=['Summary Snippet'], **{'white-space': 'normal', 'text-align': 'left'}
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # 2. FIX: Use the 'data' list (with 'Candidate Name' key) to populate the selectbox options.
    candidate_names = [cand['Candidate Name'] for cand in data]
    selected_name = st.selectbox(
        "Select a candidate to view detailed summary and outreach email:", 
        options=candidate_names
    )
    
    if selected_name:
        # Find the full data object using the selected name from the 'data' list
        selected_candidate = next(cand for cand in data if cand['Candidate Name'] == selected_name)
        st.subheader(f"Detailed Profile for {selected_name}")
        st.code(selected_candidate['Full Summary'], language='markdown')
        
        # --- Email Generation Action ---
        st.subheader("✉️ Generate Outreach Email")
        if st.button(f"Generate Email for {selected_name}", key="generate_email_btn"):
            st.session_state.email_candidate = selected_candidate
            
            with st.spinner(f"Generating personalized email for {selected_name}..."):
                # Retrieve the JD from session state to pass to the generator
                job_description_text = st.session_state.job_description_text
                
                # Find the full candidate data from the initial ranked list (raw data with lowercase keys)
                full_candidate_data = next(
                    item for item in ranked_candidates if item['name'] == selected_name
                )
                
                email_draft = generate_outreach_email(
                    job_description=job_description_text,
                    candidate_data=full_candidate_data, # Use raw data (lowercase keys) for generator
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


def run_ranking(job_description_text, top_k):
    """Handles the core RAG scoring process."""
    st.session_state.ranked_candidates = []
    
    with st.spinner("Executing Two-Stage RAG Pipeline..."):
        
        # Step 1: Run Skill Matcher (integrated within rank_candidates)
        # Step 2: Run Candidate Ranker
        ranked_candidates_list = rank_candidates(job_description_text, k=top_k)
        
        if ranked_candidates_list:
            st.session_state.ranked_candidates = ranked_candidates_list
            st.success("Ranking Complete! Results are below.")
        else:
            st.error("Ranking failed or no suitable candidates were found.")


# --- MAIN APP LAYOUT ---
def main():
    st.title("RAG Recruitment Agent")
    st.markdown("Upload a Job Description to find and rank the most suitable candidates from your existing vector index.")

    # Sidebar for Configuration
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Number of Candidates to Return (K)", min_value=1, max_value=20, value=5)
        st.markdown("**Note:** CV data is assumed to be already indexed in the `recruitment-docs` index.")

    # 1. Job Description Input (Key Input)
    st.subheader("1. Input Job Description (JD)")
    jd_file = st.file_uploader("Upload JD (.txt, .md or .pdf)", type=['txt', 'md', 'pdf'])
    
    job_description_text = ""
    if jd_file is not None:
        try:
            # Simple content reading (replace with proper PDF/DOCX parsing in a real setup)
            job_description_text = jd_file.read().decode("utf-8")
            st.session_state.job_description_text = job_description_text
            st.info("JD uploaded successfully. Ready to rank.")
        except Exception:
            st.error("Could not read file content. Please upload a plain text or markdown file for this MVP.")
    
    # Fallback to hardcoded JD for easy testing if no file is uploaded (like in your Python tests)
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
    if st.button("Run RAG Scoring Pipeline", type="primary", disabled=not job_description_text):
        run_ranking(job_description_text, top_k)

    # 3. Display Results
    if 'ranked_candidates' in st.session_state and st.session_state.ranked_candidates:
        display_ranked_candidates(st.session_state.ranked_candidates)

if __name__ == '__main__':
    main()