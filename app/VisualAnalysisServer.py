import os
import base64
import mimetypes
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("VisualAnalysisServer")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@mcp.tool()
def analyze_image(base64_image: str, mimetype: str) -> str:
    """
    Analyzes the content of a base64-encoded image and returns a description.
    """
    try:
        image_bytes = base64.b64decode(base64_image).decode('utf-8')


        prompt_text = (
            "Analyze this image in detail. Provide a concise, one-paragraph description. "
            "If it is a famous landmark, work of art, or specific location, identify it by name. "
            "Focus on the most important and defining elements in the image that would be useful for a web search. "
            "For example, instead of 'a building', say 'the Eiffel Tower in Paris'. "
            "Do not add any conversational filler; return only the description."
        )

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text}, 
                        {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,{image_bytes}"}}
                    ]
                }
            ]
        )

        description = response.choices[0].message.content

        return description

    except Exception as e:
        logging.error("Error analyzing image: %s", e)
        return f"Error analyzing image: {e}"

@mcp.tool()
def encode_image_to_base64(file_path: str) -> dict:
    """
    Encodes an image file to a base64 string along with its MIME type.
    """
    try:
        image_path = Path(file_path)
        if not image_path.is_file():
            return {"error": "File not found."}

        with open(image_path, "rb") as f:
            image_data = f.read()
        
        base64_string = base64.b64encode(image_data).decode("utf-8")
        
        mime_type, _ = mimetypes.guess_type(image_path)
        
        if not mime_type:
            mime_type = "application/octet-stream"  # default MIME type
            
        return {
            "base64_image_string": base64_string,
            "mime_type": mime_type
        }

    except FileNotFoundError:
        return {"error": f"File not found at path: {file_path}"}

    except Exception as e:
        return {"error": f"Error encoding image: {e}"}


if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run(transport="stdio")
