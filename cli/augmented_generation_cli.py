import argparse
from functools import partial
from dotenv import load_dotenv

from lib.hybrid_search import HybridSearch
from settings import PATH_MOVIES_FILE
from lib.search_utils import load_json
from lib.gemini_request import get_gemini_response


def get_search_context(query, limit=5):
    """Retrieves and formats movie data into a reusable context dictionary."""
    documents = load_json(PATH_MOVIES_FILE)["movies"]
    hybrid_search = HybridSearch(documents)
    movies = hybrid_search.rrf_search(query, limit=limit)
    
    formatted = "\n".join([
        f"Index: {i}, Title: {m['title']}, Description: {m['description']}" 
        for i, m in enumerate(movies, start=1)
    ])
    
    return {"movies": movies, "formatted_text": formatted}

def generate_rag_response(query, context, prompt_template):
    """Generates a response using the provided template and context."""
    prompt = prompt_template.format(query=query, documents=context["formatted_text"])
    return get_gemini_response(prompt)


TEMPLATES = {
    "rag": """Answer the question based on the provided documents for Hoopla users.
              Query: {query}\n\nDocuments:\n{documents}\n\nProvide a comprehensive answer:""",
    
    "summarize": """Synthesize information from multiple search results for Hoopla users.
                    Query: {query}\n\nSearch Results:\n{documents}\n\n
                    Provide a 3–4 sentence information-dense summary:""",
    
    "citations": """Answer the query for Hoopla users. Cite sources using [1], [2] format.
                    If the answer isn't in the docs, say "I don't have enough information".
                    Query: {query}\n\nDocuments:\n{documents}\n\nAnswer:""",
    
    "question": """Answer the user's question about Hoopla movies. Be casual and conversational.
                   Question: {query}\n\nDocuments:\n{documents}\n\nAnswer:"""
}

def run_command(command_type, query, limit=5):
    """A unified pipeline for executing any RAG command."""
    context = get_search_context(query, limit)
    response = generate_rag_response(query, context, TEMPLATES[command_type])

    print(f"\n--- Search Results ({command_type.upper()}) ---")
    for movie in context["movies"]:
        print(f" - {movie['title']}")
    print(f"\nResponse:\n{response}\n")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Functional RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for cmd in TEMPLATES.keys():
        p = subparsers.add_parser(cmd)
        p.add_argument("query", type=str)
        p.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.command in TEMPLATES:
        run_command(args.command, args.query, args.limit)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()