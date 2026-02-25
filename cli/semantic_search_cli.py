#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch
from settings import PATH_MOVIES_FILE
from lib.search_utils import load_json

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help='Generate and ger embedings of the query')
    embed_text_parser.add_argument("query", type=str, help="text to embed")
    
    subparsers.add_parser("verify_embeddings")

    embed_query_parser = subparsers.add_parser("embedquery")
    embed_query_parser.add_argument("query", type=str)
    
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    return parser

def embed_text_command(semantic_search: SemanticSearch, text:str ) -> None:
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings(semantic_search: SemanticSearch, load_file: str = PATH_MOVIES_FILE ):
    documents = load_json(load_file)['movies']
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(semantic_search: SemanticSearch, query: str):
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def search(semantic_search: SemanticSearch, query:str, limit: int = 5, load_file: str = PATH_MOVIES_FILE):
    documents = load_json(load_file)['movies']
    semantic_search.load_or_create_embeddings(documents)
    res = semantic_search.search(query)
    for i, movie in enumerate(res, 1):
        print(f"{i}. {movie["title"]} ({movie["score"]})")
        print(f"{movie["description"]}")

def main():
    parser = make_parser()
    args = parser.parse_args()

    semantic_search = SemanticSearch()
    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            embed_text_command(semantic_search, args.query)
        case "verify_embeddings":
            verify_embeddings(semantic_search)
        case "embedquery":
            embed_query_text(semantic_search, args.query)
        case "search":
            search(semantic_search, args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()