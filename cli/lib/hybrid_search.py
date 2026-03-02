import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from settings import PATH_CACHE_INDEX
from lib.utils import normalize_value, rrf_score


class HybridSearch():
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(PATH_CACHE_INDEX):
            self.idx.build()
            self.idx.save()

        self.doc_map = {doc["id"]: doc for doc in self.documents}

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def _normalize_scores(self, sorted_results, id_extractor, score_extractor):
        """Helper to normalize scores between 0.0 and 1.0 for a given result set."""
        if not sorted_results:
            return {}

        min_score = score_extractor(sorted_results[-1])
        max_score = score_extractor(sorted_results[0])

        if max_score == min_score:
            return {id_extractor(row): 1.0 for row in sorted_results}
        
        return {
            id_extractor(row): normalize_value(score_extractor(row), min_score, max_score)
            for row in sorted_results
        }

    def _combine_and_format_results(self, bm25_scores, semantic_scores, limit, score_combiner):
        """Helper to merge scores, sort, and map to the final output format."""
        all_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        combined_results = []

        for doc_id in all_ids:
            b_score = bm25_scores.get(doc_id, 0)
            s_score = semantic_scores.get(doc_id, 0)
            final_score = score_combiner(b_score, s_score)
            combined_results.append([doc_id, final_score, b_score, s_score])

        combined_results.sort(key=lambda x: x[1], reverse=True)

        return [{
            "id": row[0],
            "title": self.doc_map[row[0]]["title"],
            "description": self.doc_map[row[0]]["description"],
            "score": row[1],
            "bm25": row[2],
            "semantic": row[3]
        } for row in combined_results[:limit]]

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5):
        alpha = max(0.0, min(1.0, alpha))

        bm25 = self._bm25_search(query, limit * 500)
        semantic_scores = self.semantic_search.search_chunks(query, limit * 500)

        if not bm25 and not semantic_scores:
            return []

        bm25_sorted = sorted(bm25, key=lambda data: data[1], reverse=True)

        bm25_normalized = self._normalize_scores(
            bm25_sorted, id_extractor=lambda x: x[0]["id"], score_extractor=lambda x: x[1]
        )
        semantic_normalized = self._normalize_scores(
            semantic_scores, id_extractor=lambda x: x["id"], score_extractor=lambda x: x["score"]
        )

        return self._combine_and_format_results(
            bm25_normalized, 
            semantic_normalized, 
            limit, 
            score_combiner=lambda b, s: self.hybrid_score(b, s, alpha)
        )

    def rrf_search(self, query: str, k: int = 60, limit: int = 5):
        bm25 = self._bm25_search(query, limit * 500)
        semantic_scores = self.semantic_search.search_chunks(query, limit * 500)
        
        bm25_sorted = sorted(bm25, key=lambda data: data[1], reverse=True)

        bm25_rrf = {row[0]["id"]: rrf_score(i, k) for i, row in enumerate(bm25_sorted, start=1)}
        semantic_rrf = {row["id"]: rrf_score(i, k) for i, row in enumerate(semantic_scores, start=1)}

        return self._combine_and_format_results(
            bm25_rrf, 
            semantic_rrf, 
            limit, 
            score_combiner=lambda b, s: b + s
        )