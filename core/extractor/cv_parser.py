from core.utils.helpers import model
from core.utils.to_native import to_native
from google.generativeai.types import FunctionDeclaration, Tool
from typing import Dict, Any
import time

def cv_parser(text: str) -> Dict[str, Any]:
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


