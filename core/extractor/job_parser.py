from core.utils.helpers import model
from core.utils.to_native import to_native

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional,  Dict, Any
# If you are using the Gemini model, you'll need this:
from langchain_google_genai import ChatGoogleGenerativeAI


class ExtractJobDetails(BaseModel):
    """
    Extracts key details from a job description text, structured for analysis.
    """
    Job_Title: str = Field(..., description="The official title of the job position.")
    Company: str = Field(..., description="The company or organization offering the job.")
    Location: Optional[str] = Field(None, description="The city and/or country where the position is based.")
    Responsibilities: List[str] = Field(..., description="Key duties and responsibilities expected from the candidate.")
    Requirements: List[str] = Field(..., description="Essential technical and non-technical skills required for the job.")
    Preferred_Qualifications: List[str] = Field(default_factory=list, description="Additional or desired qualifications that give candidates an advantage.")
    Duration: Optional[str] = Field(None, description="The duration or contract type of the position (e.g., full-time, 3-month internship).")
    Start_Date: Optional[str] = Field(None, description="The expected or mentioned start date of the position (if available).")
    Salary_or_Benefits: Optional[str] = Field(None, description="Information about compensation or benefits, if specified.")
    Application_Deadline: Optional[str] = Field(None, description="The application deadline or closing date, if provided.")
    Keywords: List[str] = Field(default_factory=list, description="Important keywords that describe the role (skills, tools, topics).")
    Employment_Type: Optional[str] = Field(None, description="The nature of employment (e.g., Internship, Full-time, Part-time, Contract).")

    # Required fields are handled by not giving them a default value (like `Job_Title: str = Field(...)`)

def gem_json_job_langchain(job_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts structured information from a Job Description using LangChain and Gemini.
    
    Args:
        job_text (str): The full text of the job description.
    
    Returns:
        dict: Extracted structured job details.
    """
    
    # 1. Initialize the LangChain LLM (Gemini)
    # Ensure your GEMINI_API_KEY environment variable is set.
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # 2. Bind the structured output schema to the model
    # This automatically configures the LLM to use the Pydantic schema for output.
    structured_llm = model.with_structured_output(
        schema=ExtractJobDetails, 
        mode="json" 
    )

    # 3. Create the prompt for the model
    extraction_prompt = f"""
    Please analyze the following Job Description and extract all relevant details into the specified JSON format:
    ---
    {job_text}
    ---
    """
    
    # 4. Call the structured LLM
    try:
        # The invoke method handles the API call and returns a Pydantic object
        pydantic_output: ExtractJobDetails = structured_llm.invoke(extraction_prompt)
        
        # 5. Convert the Pydantic object to a standard Python dictionary
        # This gives you the full structured data dictionary immediately.
        return pydantic_output.dict()
        
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        # Return None or raise a custom exception if parsing fails
        return None

# NOTE: The explicit handling of backoff/retries was removed for conciseness 
# but can be easily re-added around the `structured_llm.invoke()` call.