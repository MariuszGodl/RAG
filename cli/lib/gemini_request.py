import os
from dotenv import load_dotenv
from google import genai


def get_gemini_response(query: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=query
    )
    return response.text