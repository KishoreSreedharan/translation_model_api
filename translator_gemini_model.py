import os
from google import genai
from google.genai import types

client = genai.Client(api_key = os.environ["GEMINI_API_KEY"])

def trnaslating(text: str) -> str:
    response = client.models.generate_content(
        model = "gemini-3.8-flash",
        contents = (
            "Translate following text into Tamil."
            "Return only in tamil language"
            "preserve the original meaning and formatting. \n"
            f"text: {text}"
        ),
        config = types.GenerateContentConfig(
            temperature = 0
        ),
        
    )
    print("Gemini response:", response)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return response.text.strip()