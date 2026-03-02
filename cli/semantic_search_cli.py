#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch
from lib.chunked_semantic_search import ChunkedSemanticSearch
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

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=5)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    subparsers.add_parser("embed_chunks")
    
    search_chunked_parser = subparsers.add_parser("search_chunked")
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)

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

def chunk(semantic_search: SemanticSearch, text:str, chunk_size: int = 200, overlap: int = 5):
    chunks= semantic_search.chunk(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")

def semantic_chunk(semantic_search: SemanticSearch, text: str, max_chunk_size: int = 4, overlap: int = 0):
    chunks = semantic_search.semantic_chunk(text, max_chunk_size, overlap )
    print(f"Semantically chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")

def embed_chunks(chunked_semantic_search: ChunkedSemanticSearch, load_file: str = PATH_MOVIES_FILE):
    documents = load_json(load_file)['movies']
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(chunked_semantic_search: ChunkedSemanticSearch, query: str, limit: int = 5):
    documents = load_json(PATH_MOVIES_FILE)['movies']
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    found_movies = chunked_semantic_search.search_chunks(query, limit)
    for i, movie in enumerate(found_movies, start=1):
        print(f"\n{i}. {movie["title"]} (score: {movie["score"]:.4f})")
        print(f"   {movie["document"]}...")
    

def main():
    parser = make_parser()
    args = parser.parse_args()

    
    match args.command:
        case "verify":
            semantic_search = SemanticSearch()
            semantic_search.verify_model()
        case "embed_text":
            semantic_search = SemanticSearch()
            embed_text_command(semantic_search, args.query)
        case "verify_embeddings":
            semantic_search = SemanticSearch()
            verify_embeddings(semantic_search)
        case "embedquery":
            semantic_search = SemanticSearch()
            embed_query_text(semantic_search, args.query)
        case "search":
            semantic_search = SemanticSearch()
            search(semantic_search, args.query, args.limit)
        case "chunk":
            semantic_search = SemanticSearch()
            chunk(semantic_search, args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_search = SemanticSearch()
            semantic_chunk(semantic_search, args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            chunked_semantic_search = ChunkedSemanticSearch()
            embed_chunks(chunked_semantic_search)
        case "search_chunked":
            chunked_semantic_search = ChunkedSemanticSearch()
            search_chunked(chunked_semantic_search, args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()