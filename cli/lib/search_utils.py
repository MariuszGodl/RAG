import json
import os
import string
from nltk.stem import PorterStemmer
from settings import *

def load_json(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def load_stop_words(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        return f.read().splitlines()

def remove_punctuation(s:str) -> str:
    table = str.maketrans('', '', string.punctuation)
    return s.translate(table)


def match_movie(movies:list, query:list) -> list:
    resoult = []
    for movie in movies:
        title = remove_punctuation(movie["title"]).lower()
        for token in query:
            if token in title:
                resoult.append(movie)
    
    return resoult


def print_movies(movies: list, nr_of_movies: int = 6) -> None:
    if len(movies) < nr_of_movies:
        nr_of_movies = len(movies)
    for i in range(nr_of_movies) :
        print(f"{i + 1}. Movie Title {movies[i]["title"]}") 

def tokenize_query(query:str) -> list[str]:
    query = remove_punctuation(query).lower()
    return query.split(sep=" ")

def remove_stop_words(words:list[str]) -> list[str]:
    stop_words = load_stop_words(PATH_STOP_WORDS)
    for q in words:
        if q in stop_words:
            words.remove(q)
    return words

def steam_the_words(words:list[str]) -> list[str]:
    stemmer = PorterStemmer()
    res = []
    for s in words:
        res.append(stemmer.stem(s))
    return res

def process_and_tokenize_text(query:str) -> list[str]:
    query = tokenize_query(query)
    query = remove_stop_words(query)
    query = steam_the_words(query)
    return query