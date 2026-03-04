import os

from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_json
from lib.utils import cosine_similarity
from settings import PATH_MOVIES_FILE


class MultimodalSearch:
    def __init__(self, documents=[], model_name="clip-ViT-B-32"):
        self.documents = documents
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])
        return image_embedding[0]

    def search_with_image(self, image_path, limit=5):
        image_embedding = self.embed_image(image_path)

        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, 2),
                }
            )

        return results


def verify_image_embedding(image_path):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path="data/paddington.jpeg", limit=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    movies = load_json(PATH_MOVIES_FILE)["movies"]
    searcher = MultimodalSearch(movies)
    results = searcher.search_with_image(image_path, limit)

    return {"image_path": image_path, "results": results}