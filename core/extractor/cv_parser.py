from core.utils.helpers import model
from core.utils.to_native import to_native
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
# If you are using the Gemini model, you'll need this:
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any
import time

#     """
#     Parses a CV string using Gemini function calling to extract structured data.
#     """
#     # Create the prompt for the model
#     extraction_prompt = f"""
#     Please analyze the following CV and extract the required information.
#     Here is the CV:
#     ---
#     {text}
#     ---
#     """

#     # Define the Function Declaration with your rich schema
#     extract_cv_details_func = FunctionDeclaration(
#         name="extract_cv_details",
#         description="Extracts key details from a CV text.",
#         parameters = {
#             "type": "object",
#             "properties": {
#                 "Name": {"type": "string", "description": "The applicant's full name"},
#                 "Contact_Info": {
#                     "type": "object",
#                     "properties": {
#                         "Email": {"type": "string"},
#                         "Phone": {"type": "string"},
#                         "LinkedIn": {"type": "string"},
#                         "Portfolio": {"type": "string"}
#                     }
#                 },
#                 "Education": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "Degree": {"type": "string"},
#                             "Major": {"type": "string"},
#                             "Institution": {"type": "string"},
#                             "Graduation_Year": {"type": "string"},
#                             "GPA": {"type": "string"}
#                         }
#                     }
#                 },
#                 "Experience": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "Title": {"type": "string"},
#                             "Company": {"type": "string"},
#                             "Duration": {"type": "string"},
#                             "Responsibilities": {"type": "string"},
#                             "Technologies": {"type": "array", "items": {"type": "string"}}
#                         }
#                     }
#                 },
#                 "Projects": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "Title": {"type": "string"},
#                             "Description": {"type": "string"},
#                             "Technologies": {"type": "array", "items": {"type": "string"}}
#                         }
#                     }
#                 },
#                 "Skills": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "Technical and soft skills (e.g., Python, Machine Learning, Communication)"
#                 },
#                 "Certifications": {"type": "array", "items": {"type": "string"}},
#                 "Languages": {"type": "array", "items": {"type": "string"}},
#                 "Career_Objective": {
#                     "type": "string",
#                     "description": "Short statement about the applicant's professional goals"
#                 },
#                 "Soft_Skills": {
#                     "type": "array",
#                     "items": {"type": "string"},
#                     "description": "Non-technical skills such as leadership, teamwork, or communication"
#                 },
#                 "Location": {
#                     "type": "string",
#                     "description": "Applicant's current city or country"
#                 },
#                 "Availability": {
#                     "type": "string",
#                     "description": "Whether the applicant is available full-time, part-time, or for internships"
#                 }
#             },
#             "required": ["Name", "Education", "Skills"]
#         }
#     )
    
#     review_tool = Tool(function_declarations=[extract_cv_details_func])

#     # Implementing exponential backoff for robustness
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             response = model.generate_content(
#                 extraction_prompt,
#                 tools=[review_tool],
#                 tool_config={'function_calling_config': 'ANY'}
#             )
#             function_call_part = response.candidates[0].content.parts[0]
#             function_call = function_call_part.function_call

#             function_args = dict(function_call.args)
#             native_data = to_native(function_args)
            
#             return native_data
            
#         except Exception as e:
#             if attempt < max_retries - 1:
#                 wait_time = 2 ** attempt
#                 # print(f"API call failed: {e}. Retrying in {wait_time}s...")
#                 time.sleep(wait_time)
#             else:
#                 print(f"LLM Extraction Error after {max_retries} attempts: {e}")
#                 return None
import time
# --- Nested Schemas ---
class ContactInfo(BaseModel):
    Email: Optional[str] = Field(None)
    Phone: Optional[str] = Field(None)
    LinkedIn: Optional[str] = Field(None)
    Portfolio: Optional[str] = Field(None)

class EducationEntry(BaseModel):
    Degree: Optional[str] = Field(None)
    Major: Optional[str] = Field(None)
    Institution: Optional[str] = Field(None)
    Graduation_Year: Optional[str] = Field(None)
    GPA: Optional[str] = Field(None)

class ExperienceEntry(BaseModel):
    Title: Optional[str] = Field(None)
    Company: Optional[str] = Field(None)
    Duration: Optional[str] = Field(None)
    Responsibilities: Optional[str] = Field(None)
    Technologies: List[str] = Field(default_factory=list)

class ProjectEntry(BaseModel):
    Title: Optional[str] = Field(None)
    Description: Optional[str] = Field(None)
    Technologies: List[str] = Field(default_factory=list)

# --- Main Schema (Replacing the FunctionDeclaration) ---
class ExtractCVDetails(BaseModel):
    """
    Extracts key details from a CV text.
    
    This docstring serves as the function's description for the LLM.
    """
    Name: str = Field(..., description="The applicant's full name")
    Contact_Info: ContactInfo = Field(..., description="Applicant's contact information")
    Education: List[EducationEntry] = Field(..., description="List of educational qualifications")
    Experience: List[ExperienceEntry] = Field(default_factory=list)
    Projects: List[ProjectEntry] = Field(default_factory=list)
    Skills: List[str] = Field(..., description="Technical and soft skills (e.g., Python, Machine Learning, Communication)")
    Certifications: List[str] = Field(default_factory=list)
    Languages: List[str] = Field(default_factory=list)
    Career_Objective: Optional[str] = Field(None, description="Short statement about the applicant's professional goals")
    Soft_Skills: List[str] = Field(default_factory=list, description="Non-technical skills such as leadership, teamwork, or communication")
    Location: Optional[str] = Field(None, description="Applicant's current city or country")
    Availability: Optional[str] = Field(None, description="Whether the applicant is available full-time, part-time, or for internships")

# The required fields ("Name", "Education", "Skills") are automatically inferred 
# because they don't have a default value or are explicitly marked with `...`.
def cv_parser(text: str) -> Optional[Dict[str, Any]]:
    """
    Parses a CV string using LangChain and Gemini for structured data extraction.
    """
    # 1. Initialize the LangChain LLM (Gemini)
    # The API key will be read from the GEMINI_API_KEY environment variable.
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # 2. Bind the structured output schema to the model
    # This tells the model to ONLY return data conforming to the ExtractCVDetails class.
    # The response is automatically parsed into a Pydantic object.
    structured_llm = model.with_structured_output(
        schema=ExtractCVDetails, 
        # For LangChain, this mode is often sufficient and replaces the tool_config
        mode="json" 
    )

    # 3. Create the prompt
    extraction_prompt = f"""
    Please analyze the following CV and extract the required information into the structured JSON format.
    Here is the CV:
    ---
    {text}
    ---
    """
    
    # 4. Implement exponential backoff for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Invoke the structured LLM with the prompt
            pydantic_output: ExtractCVDetails = structured_llm.invoke(extraction_prompt)
            
            # 5. Convert the Pydantic object to a standard Python dictionary
            return pydantic_output.dict()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"LLM call failed on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"LLM Extraction Error after {max_retries} attempts: {e}")
                return None

# --- Example Usage ---
# sample_cv_text = "John Doe, 123 Main St, john@example.com, M.Sc. Computer Science from MIT (2020), Skills: Python, AWS."
# extracted_data = cv_parser_langchain(sample_cv_text)
# print(extracted_data)

