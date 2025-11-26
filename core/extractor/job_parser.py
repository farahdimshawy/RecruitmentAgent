from core.utils.helpers import model
from core.utils.to_native import to_native

from google.generativeai.types import FunctionDeclaration, Tool
from google.protobuf.json_format import MessageToDict
import json

def gem_json_job(job_text):
    """
    Extracts structured information from a Job Description using Gemini function calling.
    
    Args:
        job_text (str): The full text of the job description.
        model: The Gemini model instance (e.g., genai.GenerativeModel).
    
    Returns:
        dict: Extracted structured job details (title, company, requirements, etc.)
    """

    # Use your existing job extraction tool
    extract_job_details_func = FunctionDeclaration(
    name="extract_job_details",
    description="Extracts key details from a job description text.",
    parameters={
        "type": "object",
        "properties": {
            "Job_Title": {
                "type": "string",
                "description": "The official title of the job position."
            },
            "Company": {
                "type": "string",
                "description": "The company or organization offering the job."
            },
            "Location": {
                "type": "string",
                "description": "The city and/or country where the position is based."
            },
            "Responsibilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key duties and responsibilities expected from the candidate."
            },
            "Requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Essential technical and non-technical skills required for the job."
            },
            "Preferred_Qualifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional or desired qualifications that give candidates an advantage."
            },
            "Duration": {
                "type": "string",
                "description": "The duration or contract type of the position (e.g., full-time, 3-month internship)."
            },
            "Start_Date": {
                "type": "string",
                "description": "The expected or mentioned start date of the position (if available)."
            },
            "Salary_or_Benefits": {
                "type": "string",
                "description": "Information about compensation or benefits, if specified."
            },
            "Application_Deadline": {
                "type": "string",
                "description": "The application deadline or closing date, if provided."
            },
            "Keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important keywords that describe the role (skills, tools, topics)."
            },
            "Employment_Type": {
                "type": "string",
                "description": "The nature of employment (e.g., Internship, Full-time, Part-time, Contract)."
            },
        },
        "required": ["Job_Title", "Company", "Responsibilities", "Requirements"]
    }
)

    review_tool = Tool(function_declarations=[extract_job_details_func])

    # Create the prompt for Gemini
    extraction_prompt = f"""
    Please analyze the following Job Description and extract all relevant details such as:
    - Job Title
    - Company
    - Location
    - Responsibilities
    - Requirements
    - Preferred Qualifications
    - Duration (if internship)
    ---
    {job_text}
    ---
    """

    # Call Gemini API
    response = model.generate_content(
        extraction_prompt,
        tools=[review_tool],
        tool_config={"function_calling_config": "ANY"}
    )
    function_call_part = response.candidates[0].content.parts[0]
    function_call = function_call_part.function_call

    # Convert the MapComposite into a normal Python dict
    function_args = dict(function_call.args)
    native_data = to_native(function_args)


    # Safely access values
    extracted_data = {
        'Job_Title': native_data.get('Job_Title'),
        'Company': native_data.get('Company'),
        'Location': native_data.get('Location'),
        'Responsibilities': native_data.get('Responsibilities'),
        'Requirements': native_data.get('Requirements'),
        'Preferred_Qualifications': native_data.get('Preferred_Qualifications'),
        'Duration': native_data.get('Duration'),
    }

    # Convert to native Python types (if using protobuf types)

    return extracted_data
