from lib.semantic_search import SemanticSearch
from lib.utils import cosine_similarity
from settings import *
import numpy as np
import json
import os

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.documents = None
        self.documents_map = {}

    def build_chunk_embeddings(self, documents: list) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(self.documents):
            text = doc.get("description", "")
            if not text:
                continue
            
            semantic_chunks = self.semantic_chunk(text, 4, 1)
            total_chunks = len(semantic_chunks)

            for chunk_idx, chunk_text in enumerate(semantic_chunks):
                all_chunks.append(chunk_text)
                
                chunk_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        with open(PATH_CACHE_CHUNK_EMBEDDINGS, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(PATH_CACHE_CHUNK_METADATA, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list:
        query_embedding = self.generate_embedding(query)
        chunk_score = []
        for idx, meta in enumerate(self.chunk_metadata["chunks"], start=0):
            chunk_score.append({
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": cosine_similarity(query_embedding, self.chunk_embeddings[idx])
            })

        score_map = {}
        for row in chunk_score:
            movie_idx, score = row["movie_idx"], row["score"] 
            if not row["movie_idx"] in score_map:
                score_map[movie_idx] = score
                continue
            if score > score_map[movie_idx]:
                score_map[movie_idx] = score
        
        sorted_movies = sorted(score_map.items(), key=lambda data: data[1], reverse=True)[:limit]
        return [{
            "id": self.documents[movie_idx]["id"],
            "title": self.documents[movie_idx]["title"],
            "document": self.documents[movie_idx]["description"][:100],
            "score": round(score, 3),
            "metadata": self.documents[movie_idx].get("metadata", {})
            } for movie_idx, score in sorted_movies]

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if os.path.exists(PATH_CACHE_CHUNK_EMBEDDINGS) and os.path.exists(PATH_CACHE_CHUNK_METADATA):
            with open(PATH_CACHE_CHUNK_EMBEDDINGS, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(PATH_CACHE_CHUNK_METADATA, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
