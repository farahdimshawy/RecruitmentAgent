import fitz  

def pdf_extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF (fitz).
    Requires the 'PyMuPDF' library to be installed.
    """
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text


