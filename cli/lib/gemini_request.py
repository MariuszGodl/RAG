import os
from dotenv import load_dotenv
from google import genai
import json
from sentence_transformers import CrossEncoder
import numpy as np

def get_gemini_response(query: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=query
    )
    return response.text

def evaluate_results(movies: list, query:str):
    formatted_movies = [ f"Index: {i}, Title: {movie["title"]}, Description: {movie["description"]}" for i,  movie in enumerate(movies, start=1)]
    agent_query = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {chr(10).join(formatted_movies)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""
    response = get_gemini_response(agent_query)
    scores = json.loads(response) 
    for i, (movie, score) in enumerate(zip(movies, scores), start=1):
        print(f"{i}. {movie["title"]} {score}/3")

def rerank_docs(docs: list[dict], rerank_type:str, query:str, limit: int = 5):
    
    def individual_query(doc):
        return f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""

    def batch_query(batch_doc_str):
        return f"""Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies:
            {batch_doc_str}

            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]
            """

    match rerank_type:
        case "individual":
            scored_docs =  [ {
                **doc,
                "Rerank" : float(get_gemini_response(individual_query(doc)))
                        } for doc in docs ]
            sorted_docs = sorted(scored_docs, key=lambda data: data["Rerank"], reverse=True)
        case "batch":
            doc_list_str = [ f"ID: {str(doc.get("id"))},  Title {doc.get("title")}, Description: {doc.get("description")}" for doc in docs ]
            doc_str = "\n".join(doc_list_str)
            response = get_gemini_response(batch_query(doc_str))
            
            scores = json.loads(response)
            scored_docs = [{
                **doc,
                "Rerank": scores.index(doc["id"]) + 1
            } for doc in docs ]
            sorted_docs = sorted(scored_docs, key=lambda data: data["Rerank"])
        case "cross_encoder":
            pairs = []
            for doc in docs:
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            final_docs = []
            for (doc, score) in scored_docs:
                final_docs.append({
                    **doc,
                    "CrossEncoder": float(score) # Optional: keep the score for debugging
                })
            
            return final_docs[:limit]
    return sorted_docs[:limit]


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