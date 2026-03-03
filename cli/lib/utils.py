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

def normalize(values: list[float]):
    if len(values) == 0: return []
    low = min(values)
    high = max(values)
    if low == high: return [ 1 ] * len(values)
    return [ (value - low) / (high - low) for value in values ]

def normalize_value(value: float, low: float, high: float):
    if low == high: return 1 
    return (value - low) / (high - low)

def rrf_score(rank, k=60):
    return 1 / (k + rank)





def enhance_query(query: str, enhance: str = None) -> str:
    def spell_enhance(query: str) -> str:
        return f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:"""

    def rewrite_enhance(query: str) -> str:
        return f"""Rewrite this movie search query to be more specific and searchable.

            Original: "{query}"

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep it concise (under 10 words)
            - It should be a google style search query that's very specific
            - Don't use boolean logic

            Examples:

            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            Rewritten query:"""

    def expand_enhance(query: str):
        return f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """
    
    if enhance == "spell":
        return spell_enhance(query)
    elif enhance == "rewrite":
        return rewrite_enhance(query)
    elif enhance == "expand":
        return expand_enhance(query)
    
    return query