import pickle
import numpy as np

def load_binary(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required index file missing: {path}")
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(f"Cache file at {path} is corrupted.") from e

def save_binary(content, path: str):
    try:
        with open(path, "wb") as f:
            pickle.dump(content, f )
    except FileNotFoundError:
        raise FileNotFoundError(f"Required index file missing: {path}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
