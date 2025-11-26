from core.utils.helpers import model
import os
import fitz  # PyMuPDF

def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_all_pdfs(folder_path):
    """Extract text from all PDFs in a folder and return a list of strings."""
    pdf_texts = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                text = extract_text(pdf_path)
                pdf_texts.append(text)
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
    
    return pdf_texts
