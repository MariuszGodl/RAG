from pathlib import Path
from settings import BM25_B, BM25_K1,PATH_CACHE_DOC_LENGTH, PATH_CACHE_TERM_FREQUENCIES, PATH_MOVIES_FILE, PATH_CACHE_INDEX, PATH_CACHE_DOCMAP, PATH_CACHE
from lib.search_utils import normalize_and_tokenize_query, load_json
from lib.utils import load_binary, save_binary
import pickle
import os
from collections import Counter
import math

class InvertedIndex():
    """In-memory inverted index mapping tokens to document IDs and storing document metadata."""

    def __init__(self):
        """Initialize an empty index and document map.

        index: dict mapping token (str) -> set of document IDs (int).
        docmap: dict mapping document ID (int) -> full document object (dict).
        """
        self.index = dict() # mapping tokens (strings) to sets of document IDs (integers).
        self.docmap = dict() #  mapping document IDs to their full document objects.
        self.term_frequencies = dict() # mapping document IDs to Counter objects
        self.doc_lengths = dict() # mapping dockument IDs to its lenght
    def get_index(self) -> dict:
        """Return the internal index structure.

        Returns:
        - dict: token -> set(document IDs)
        """
        return self.index

    def get_docmap(self) -> dict:
        return self.docmap

    def get_doc_by_id(self, id: int) -> dict:
        return self.docmap[id]

    def get_docs_by_ids(self, ids: list) -> list[dict]:
        result = []
        ids = sorted(ids)
        for id in ids:
            result.append(self.get_doc_by_id(id))
        return result

    def get_movies_form_token(self, token: str) -> list[dict]:
        if token in self.index:
            return self.get_doc_by_id(self.index[token])

    def __add_document(self, doc_id:int, text:str):
        """Tokenize the input text and add each token to the index with the document ID.

        Parameters:
        - doc_id: Identifier of the document to add.
        - text: Text content to tokenize and index.
        """
        processed_text = normalize_and_tokenize_query(text)

        self.doc_lengths[doc_id] = len(processed_text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter(processed_text)

        for token in processed_text:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            
    def get_documents(self, term:str) -> list[int]: 
        """Get document IDs for a given token.

        Parameters:
        - term: Token string to look up.

        Returns:
        - list[int]: List of document IDs (sorted ascending if available).
        """
        return sorted(self.index.get(term, set()))
    
    def get_average_doc_lenght(self) -> int:
        res = 0
        for val in self.doc_lengths.values():
            res += val
        return res / len(self.doc_lengths)

    @classmethod
    def extract_single_token(cls, token:str) -> str:
        tokens = normalize_and_tokenize_query(token)
        if len(tokens) != 1:
            raise ValueError(
                f"Expected 1 token after normalization, but found {len(tokens)}."
            )
        return tokens[0]

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term] if doc_id in self.term_frequencies else 0

    def get_idf(self, term:str) -> float:
        term_match_doc_count = len(self.get_documents(term))
        total_doc_count = len(self.get_docmap())
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
 
    def get_tfidf(self, doc_id:int, term:str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        term = InvertedIndex.extract_single_token(term)
        N = len(self.get_docmap())
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:

        tf = self.get_tf(doc_id, term)
        length_norm = 1 - BM25_B + BM25_B * (self.doc_lengths[doc_id] / self.get_average_doc_lenght())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm) 

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> list:
        tokens = normalize_and_tokenize_query(query)
        scores = {} # DocId -> score
        for token in tokens:
            doc_ids = self.index.get(token, None)
            if doc_ids is not None:
                for doc_id in doc_ids:
                    scores[doc_id] = scores.get(doc_id, 0) + self.bm25(doc_id, token)
        sorted_doc = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]
        movies = self.get_docs_by_ids([doc_id for doc_id, score in sorted_doc])
        return [ [movie, scores[movie["id"]]] for movie in movies]

    def build(self) -> None:
        """Build the index and document map from the movies JSON.

        Reads movies from PATH_MOVIES_FILE and indexes each movie's title and description.
        """
        movies = load_json(PATH_MOVIES_FILE)["movies"]
        for movie in movies:

            self.__add_document(movie.get("id", ""), movie["title"] + " " + movie["description"] )
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        """Persist the index and docmap to disk using pickle."""
        Path(PATH_CACHE).mkdir(parents=True, exist_ok=True)

        save_binary(self.index, PATH_CACHE_INDEX)
        save_binary(self.docmap, PATH_CACHE_DOCMAP)
        save_binary(self.term_frequencies, PATH_CACHE_TERM_FREQUENCIES)
        save_binary(self.doc_lengths, PATH_CACHE_DOC_LENGTH)

    def load(self) -> None:
        if os.path.exists(PATH_CACHE):
            self.index = load_binary(PATH_CACHE_INDEX)
            self.docmap = load_binary(PATH_CACHE_DOCMAP)
            self.term_frequencies = load_binary(PATH_CACHE_TERM_FREQUENCIES)
            self.doc_lengths = load_binary(PATH_CACHE_DOC_LENGTH)
        else:
            raise FileNotFoundError("Path of the cache does not exists.")