from pathlib import Path
from settings import PATH_MOVIES_FILE, PATH_CACHE_INDEX, PATH_CACHE_DOCMAP, PATH_CACHE
from lib.search_utils import normalize_and_tokenize_query, load_json
import pickle
import os
from settings import *

class InvertedIndex():
    """In-memory inverted index mapping tokens to document IDs and storing document metadata."""

    def __init__(self):
        """Initialize an empty index and document map.

        index: dict mapping token (str) -> set of document IDs (int).
        docmap: dict mapping document ID (int) -> full document object (dict).
        """
        self.index = dict() # mapping tokens (strings) to sets of document IDs (integers).
        self.docmap = dict() #  mapping document IDs to their full document objects.

    def get_index(self) -> dict:
        """Return the internal index structure.

        Returns:
        - dict: token -> set(document IDs)
        """
        return self.index

    def __add_document(self, doc_id:int, text:str):
        """Tokenize the input text and add each token to the index with the document ID.

        Parameters:
        - doc_id: Identifier of the document to add.
        - text: Text content to tokenize and index.
        """
        processed_text = normalize_and_tokenize_query(text)
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
        with open(PATH_CACHE_INDEX, "wb") as f:
            pickle.dump(self.index, f)

        with open(PATH_CACHE_DOCMAP, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        if os.path.exists(PATH_CACHE):
            if os.path.exists(PATH_CACHE_INDEX):
                with open(PATH_CACHE_INDEX, "rb") as f:
                    self.index = pickle.load(f)
            else:
                raise Exception("Path of cache index does not exists.")

            if os.path.exists(PATH_CACHE_DOCMAP):
                with open(PATH_CACHE_DOCMAP, "rb") as f:
                    self.docmap = pickle.load(f)
            else:
                raise Exception("Path of cache docmap does not exists.")
        else:
            raise Exception("Path of the cache does not exists.")