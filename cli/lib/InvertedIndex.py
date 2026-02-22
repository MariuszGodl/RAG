from settings import *
from lib.search_utils import *
import pickle

class InvertedIndex():

    def __init__(self):
        self.index = dict() # mapping tokens (strings) to sets of document IDs (integers).
        self.docmap = dict() #  mapping document IDs to their full document objects.

    def get_index(self) -> dict:
        return self.index

    def __add_document(self, doc_id:int, text:str):
        """Tokenize the input text, then add each token to the index with the document ID."""
        processed_text = process_and_tokenize_text(text)
        for token in processed_text:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
    
    def get_documents(self, term:str) -> list[int]: 
        """Get the set of document IDs for a given token, and return them as a list, sorted in ascending order. """
        return self.index.get(term, []).sort()

    def build(self) -> None:
        """iterate over all the movies and add them to both the index and the docmap"""
        movies = load_json(PATH_MOVIES_FILE)["movies"]
        for movie in movies:

            self.__add_document(movie.get("id", ""), movie["title"] + " " + movie["description"] )
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        """save the index and docmap attributes to disk using the pickle"""
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)