import argparse
from lib.search_utils import print_movie_list_with_scores, normalize_and_tokenize_query, load_json, match_movies_by_title, print_movie_list
from lib.keyword_search import InvertedIndex
from settings import PATH_MOVIES_FILE, BM25_K1, BM25_B
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

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
        )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    return parser


def search_command(i_index: InvertedIndex, args: argparse.Namespace) -> None:
    print(f"Searching for: {args.query}")
    query = normalize_and_tokenize_query(args.query)
    result = match_movies_by_title(i_index, query)
    print_movie_list(result)

def tf_command(i_index: InvertedIndex, args: argparse.Namespace) -> None: 
    doc_id = args.doc_id
    term = InvertedIndex.extract_single_token(args.term)
    print(i_index.get_tf(doc_id, term)) 

def idf_command(i_index: InvertedIndex, args: argparse.Namespace) -> None:
    term = InvertedIndex.extract_single_token(args.term)
    idf = i_index.get_idf(term)
    print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

def tfidf_command(i_index: InvertedIndex, args: argparse.Namespace) -> None:
    term = InvertedIndex.extract_single_token(args.term)
    tf_idf = i_index.get_tfidf(args.doc_id, term)
    print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

def bm25_idf_command(i_index: InvertedIndex, args: argparse.Namespace) -> None:
    term = InvertedIndex.extract_single_token(args.term)
    bm25_idf =  i_index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

def bm25_tf_command(i_index: InvertedIndex, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> None:
    term = InvertedIndex.extract_single_token(term)
    bm25tf = i_index.get_bm25_tf(doc_id, term, k1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

def bm25_search(i_index: InvertedIndex, query:str) -> None:
    res = i_index.bm25_search(query)
    print_movie_list_with_scores(res)

def main() -> None:

    parser = make_parsers()
    args = parser.parse_args()
    i_index = InvertedIndex()
    match args.command:
        case "build":
            i_index.build()
            i_index.save()
        case _:
            i_index.load()
            match args.command:
                case "search":
                    search_command(i_index, args)
                case "tf":
                    tf_command(i_index, args)
                case "idf":
                    idf_command(i_index, args)
                case "tfidf":
                    tfidf_command(i_index, args)
                case "bm25idf":   
                    bm25_idf_command(i_index, args)
                case "bm25tf":
                    bm25_tf_command(i_index, args.doc_id, args.term, args.k1, args.b)
                case "bm25search":
                    bm25_search(i_index, args.query)
                case _:
                    parser.print_help()


if __name__ == "__main__":
    main()
