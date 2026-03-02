import argparse
from lib.utils import normalize
from lib.hybrid_search import HybridSearch
from settings import PATH_MOVIES_FILE
from lib.search_utils import load_json

def create_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize")
    normalize_parser.add_argument("values", type=float, nargs="+")
    
    weighted_search_parser = subparser.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    rrf_search_parser = subparser.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)

    return parser

def normalize_command(values: list[int]):
    normalized_values = normalize(values)
    [print(f"* {val:.4f}") for val in normalized_values]


def weighted_search_command(query: str, alpha: float = 0.5, limit: int = 5, load_file: str = PATH_MOVIES_FILE):
    documents = load_json(load_file)['movies']
    hybrid_search = HybridSearch(documents)
    movies = hybrid_search.weighted_search(query, alpha, limit)

    for i, movie in enumerate(movies, start=1):
        print(f"{i}. {movie["title"]}")
        print(f"Hybrid Score: {movie["score"]}")
        print(f"BM25: {movie["bm25"]} , Semantic: {movie["semantic"]}")
        print(f"{movie["description"]}")

def rrf_search(query: str, k: int = 60, limit:int = 5, load_file: str = PATH_MOVIES_FILE):
    documents = load_json(load_file)['movies']
    hybrid_search = HybridSearch(documents)
    movies = hybrid_search.rrf_search(query, k, limit)

    for i, movie in enumerate(movies, start=1):
        print(f"{i}. {movie["title"]}")
        print(f"RRF Score: {movie["score"]:.2f}")
        print(f"BM25 Rank: {movie["bm25"]} , Semantic Rank: {movie["semantic"]}")
        print(f"{movie["description"][:50]}")

def main() -> None:

    parser = create_parsers()
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.values)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()