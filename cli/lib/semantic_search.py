from sentence_transformers import SentenceTransformer
import numpy as np
import os 
from pathlib import Path
from settings import PATH_CACHE, PATH_CACHE_EMBEDDINGS
from lib.search_utils import strip_punctuation
from lib.utils import cosine_similarity

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def verify_model(self) -> None:
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

    def generate_embedding(self, text: str):
        text = strip_punctuation(text.strip().lower())
        if text is None or text == "":
            raise ValueError()
        return self.model.encode([text])[0]

    def search(self, query: str, limit: int = 5):
        if len(self.embeddings) != len(self.documents) and not self.embeddings:
            ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        cosine_sim_res = []
        for idx, doc_embedding in enumerate(self.embeddings, 1):
            cosine_sim_res.append((cosine_similarity(query_embedding, doc_embedding), self.documents_map[idx] ))
        cosine_sim_res_sorted = sorted(cosine_sim_res, key=lambda data: data[0], reverse=True)[:limit]
        return [ {"score": data[0],
           "title": data[1]["title"],
           "description": data[1]["description"]
           }
          for data in cosine_sim_res_sorted ]
        
        

    def build_embeddings(self, documents: list):
        self.documents = documents
        string_doc = []
        for doc in documents:
            self.documents_map[doc["id"]] = doc
            string_doc.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(string_doc, show_progress_bar=True)
        Path(PATH_CACHE).mkdir(parents=True, exist_ok=True)
        with open(PATH_CACHE_EMBEDDINGS, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list):
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc
        if os.path.exists(PATH_CACHE_EMBEDDINGS):
            with open(PATH_CACHE_EMBEDDINGS, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)
         

    def get_model(self):
        return self.model