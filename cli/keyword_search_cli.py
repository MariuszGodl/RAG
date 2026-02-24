import argparse
from lib.search_utils import normalize_and_tokenize_query, load_json, match_movies_by_title, print_movie_list
from lib.InvertedIndex import InvertedIndex
from settings import PATH_MOVIES_FILE
import math

def make_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index")
    
    term_parser = subparsers.add_parser("tf", help="Get frequency of given term for given doc id")
    term_parser.add_argument("doc_id",type=int, help="Doc Id")
    term_parser.add_argument("term", type=str, help="Word to count")
    
    idf_parser = subparsers.add_parser("idf", help="measures how many documents in the dataset contain a term")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="get tfidf for given doc and term")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)
    return parser

def main() -> None:

    parser = make_parsers()
    args = parser.parse_args()
    i_index = InvertedIndex()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            i_index.load()
            query = normalize_and_tokenize_query(args.query)
            result = match_movies_by_title(i_index, query)
            print_movie_list(result)
            
            return result

        case "tf":
            i_index.load()
            doc_id = args.doc_id
            term = normalize_and_tokenize_query(args.term)[0]
            print(i_index.get_tf(doc_id, term)) 

        case "idf":
            i_index.load()
            term = normalize_and_tokenize_query(args.term)[0]
            idf = i_index.get_idf(term)
            print(f"Inverse document frequency of '{args._get_kwargsterm}': {idf:.2f}")

        case "tfidf":
            i_index.load()
            term = normalize_and_tokenize_query(args.term)[0]
            tf_idf = i_index.get_tfidf(args.doc_id, term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "build":
            i_index.build()
            i_index.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
