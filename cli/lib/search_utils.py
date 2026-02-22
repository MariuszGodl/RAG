import json
import os
import string
from pathlib import Path
from nltk.stem import PorterStemmer
from typing import List, Dict
from settings import PATH_STOP_WORDS

def load_json(file_path: str) -> list[dict]:
	"""Load and parse a JSON file.

	Parameters:
	- file_path: Path to the JSON file.

	Returns:
	- A list of dicts parsed from the JSON file. Returns an empty list if file not found.
	"""
	if not os.path.exists(file_path):
		print(f"Error: {file_path} not found.")
		return []
	with open(file_path, "r") as f:
		return json.load(f)

def load_stop_words(file_path: str) -> List[str]:
	"""Read stop words from a file, one per line.

	Parameters:
	- file_path: Path to the stop words file.

	Returns:
	- A list of stop words (strings). Returns empty list on error.
	"""
	try:
		p = Path(file_path)
		if not p.exists():
			return []
		return p.read_text(encoding="utf-8").splitlines()
	except Exception:
		return []

def strip_punctuation(s: str) -> str:
	"""Remove punctuation characters from a string.

	Parameters:
	- s: Input string.

	Returns:
	- The string with punctuation removed.
	"""
	table = str.maketrans('', '', string.punctuation)
	return s.translate(table)


def match_movies_by_title(movies: list, tokens: list) -> list:
	"""Find movies whose title contains any token from the tokens list.

	Parameters:
	- movies: List of movie dicts (each expected to have a 'title' key).
	- tokens: List of tokens (strings) to match against movie titles.

	Returns:
	- List of movie dicts that matched at least one token.
	"""
	results = []
	for movie in movies:
		title = strip_punctuation(movie["title"]).lower()
		for token in tokens:
			if token in title:
				results.append(movie)
	
	return results


def print_movie_list(movies: list, nr_of_movies: int = 6) -> None:
	"""Print a numbered subset of movies to stdout.

	Parameters:
	- movies: List of movie dicts (each expected to have a 'title' key).
	- nr_of_movies: Maximum number of movies to print (default 6).
	"""
	if len(movies) < nr_of_movies:
		nr_of_movies = len(movies)
	for i in range(nr_of_movies):
		print(f"{i + 1}. Movie Title {movies[i]['title']}") 

def tokenize_text(text: str) -> list[str]:
	"""Lowercase, remove punctuation, and split a text string into tokens.

	Parameters:
	- text: Raw text or query string.

	Returns:
	- List of token strings.
	"""
	text = strip_punctuation(text).lower()
	return text.split()

def remove_stop_tokens(tokens: list[str]) -> list[str]:
	"""Remove stop tokens from a list of tokens using PATH_STOP_WORDS.

	Parameters:
	- tokens: List of token strings.

	Returns:
	- Filtered list with stop tokens removed.
	"""
	stop_words = set(load_stop_words(PATH_STOP_WORDS))
	return [t for t in tokens if t not in stop_words]

def stem_tokens(tokens: list[str]) -> list[str]:
	"""Apply Porter stemming to a list of tokens.

	Parameters:
	- tokens: List of token strings.

	Returns:
	- List of stemmed token strings.
	"""
	stemmer = PorterStemmer()
	return [stemmer.stem(t) for t in tokens]

def normalize_and_tokenize_query(query: str) -> list[str]:
	"""Tokenize, remove stop tokens, and stem a query string.

	Parameters:
	- query: Raw query string.

	Returns:
	- List of processed tokens ready for matching.
	"""
	tokens = tokenize_text(query)
	tokens = remove_stop_tokens(tokens)
	tokens = stem_tokens(tokens)
	return tokens
