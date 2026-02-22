import argparse
from lib.search_utils import *
from settings import *
from lib.InvertedIndex import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_paser = subparsers.add_parser("build", help="Build Inverted Index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            query = process_and_tokenize_text(args.query)
            
            
            movies = load_json(PATH_MOVIES_FILE)["movies"]
            
            result = match_movie(movies, query)
            
            print_movies(result)
            
            return result
        case "build":
                i_index = InvertedIndex()
                i_index.build()
                i_index.save()
                index = i_index.get_index()
                print(sorted(index["merida"])[0])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
