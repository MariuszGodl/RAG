import argparse
from lib.search_utils import normalize_and_tokenize_query, load_json, match_movies_by_title, print_movie_list
from lib.InvertedIndex import InvertedIndex
from settings import PATH_MOVIES_FILE

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            i_index = InvertedIndex()
            i_index.load()

            query = normalize_and_tokenize_query(args.query)
            
            movies = load_json(PATH_MOVIES_FILE)["movies"]
            
            result = match_movies_by_title(movies, query)
            
            print_movie_list(result)
            
            return result
        case "build":
                i_index = InvertedIndex()
                i_index.build()
                i_index.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
