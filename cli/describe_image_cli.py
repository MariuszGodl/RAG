import argparse
import json
import mimetypes
from dotenv import load_dotenv
from google.genai import types

# Assuming these are from your local modules
from settings import PATH_GOLDEN_JSON, PATH_MOVIES_FILE
from lib.search_utils import load_json
from lib.gemini_request import get_gemini_response
from lib.multimodal_search import MultimodalSearch

def create_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal Search & Evaluation CLI")

    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--query", type=str, required=True, help="The text query")

    return parser


def verify_image_embedding(image_path: str):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def describe_image(image_path: str, query: str):
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    parts = [
        system_prompt,
        types.Part.from_bytes(data=image_bytes, mime_type=mime),
        query.strip(),
    ]    
    
    response = get_gemini_response(parts, False)
    print(f"Rewritten query: {response.text.strip()}")
    
    if getattr(response, 'usage_metadata', None) is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


def main():
    load_dotenv()
    parser = create_parsers()
    args = parser.parse_args()

    describe_image(args.image, args.query)


if __name__ == "__main__":
    main()