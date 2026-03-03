import argparse
import json
from settings import PATH_GOLDEN_JSON, PATH_MOVIES_FILE
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_json

def create_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    return parser

def main():

    parser = create_parsers()
    args = parser.parse_args()
    limit = args.limit
    with open(PATH_GOLDEN_JSON, "r") as f:
        golden_tests = json.load(f)["test_cases"]
    
    docs = load_json(PATH_MOVIES_FILE)["movies"]

    hybrid_search = HybridSearch(docs)
    test_results = []
    for test_case in golden_tests:
        results = hybrid_search.rrf_search(test_case["query"],k=60, limit=limit)
        relevant_retrieved = 0

        for res in results:
            if res["title"] in test_case["relevant_docs"]:
                relevant_retrieved += 1

        total_retrieved = len(results)
        precision = (
            relevant_retrieved / total_retrieved
            if total_retrieved > 0
            else 0
        )

        total_relevant = len(test_case["relevant_docs"])
        recall = (
            relevant_retrieved / total_relevant
            if total_relevant > 0
            else 0
        )

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        test_results.append({
            "query" : test_case["query"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrived_titles": ", ".join([doc["title"] for doc in results]),
            "relevant_titles": ", ".join(test_case["relevant_docs"])
        } )
    for test_res in test_results:
        print(f"- Query: {test_res["query"]}")
        print(f"    - Precision@{limit}: {test_res["precision"]:.4f} ") 
        print(f"    - Recall@{limit}: {test_res["recall"]:.4f}")
        print(f"    - F1 Score: {test_res["f1"]:.4f}")
        print(f"    - Retrieved: {test_res["retrived_titles"]} ") 
        print(f"    - Relevant: {test_res["relevant_titles"]} ") 


if __name__ == "__main__":
    main()