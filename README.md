# Retrieval-Augmented-Generation (RAG) Search Engine

This repository contains a small, experimental search engine built around a
variety of information retrieval techniques and the Google GenAI API.  It can
be used as a foundation for prototyping retrieval‑augmented generation (RAG)
workflows against a simple dataset of movies, but the core logic is totally
generic and can be repurposed for other domains.

## 🚀 Features

- **Keyword search** using an inverted index and BM25 ranking.
- **Semantic search** powered by `sentence-transformers` embeddings.
- **Hybrid retrieval** combining lexical and vector distances with RRF.
- **Multimodal search** using a CLIP‑style model to match images with text.
- **Augmented generation CLI** that fetches relevant documents and prompts
  Google GenAI to answer, summarize, or cite sources.
- **Utility scripts** for dataset inspection, evaluation, and image
  description.
- Caching of intermediate results (embeddings, indexes) under `cache/`.

## 📁 Project Structure

```
.
├── cli/                     # command‑line entrypoints
│   ├── augmented_generation_cli.py
│   ├── semantic_search_cli.py
│   ├── keyword_search_cli.py
│   ├── hybrid_search_cli.py
│   ├── multimodal_search_cli.py
│   ├── describe_image_cli.py
│   ├── evaluation_cli.py
│   └── lib/                 # reusable library modules
│       ├── semantic_search.py
│       ├── keyword_search.py
│       ├── hybrid_search.py
│       ├── multimodal_search.py
│       ├── search_utils.py
│       ├── utils.py
│       └── gemini_request.py
├── data/                    # sample data files
│   ├── movies.json          # collection of movie documents
│   ├── stopwords.txt        # training set for tokenization
│   └── golden_dataset.json  # used for evaluation
├── cache/                   # automatically populated at runtime
├── pyproject.toml           # pip/Poetry metadata and dependencies
└── README.md                # this file
```

> **Note:** the CLI tools all load the JSON dataset defined in
> `cli/settings.py` and write caches there as well.

## 🛠️ Installation

1. Clone the repository and navigate to the root directory:
   ```bash
   git clone <repo-url> rag-search-engine
   cd rag-search-engine
   ```

2. Create a Python environment (venv, conda, etc.) and activate it.

3. Install dependencies:
   ```bash
   pip install -r <(python - <<'PY'
import tomllib, sys
print("\n".join(tomllib.load(open('pyproject.toml'))['project']['dependencies']))
PY
)
   ```

   or simply:

   ```bash
   pip install -e .
   ```

4. (Optional) set environment variables via a `.env` file if you plan to
   exercise the augmented generation commands.  A valid Google GenAI API key
   should be stored in `GOOGLE_API_KEY`.

## 📚 Data

The only dataset shipped with the project is `data/movies.json`, which holds a
list of movie records—each with an `id`, `title`, and `description`.  You can
replace this file with your own collection, as long as the same structure is
maintained.  Stopwords for keyword indexing are read from
`data/stopwords.txt`.

## ⚙️ Usage Examples

Most of the functionality is exposed via CLI scripts.  Run them from the project
root, e.g.:

```bash
python -m cli.keyword_search_cli build        # build inverted index
python -m cli.keyword_search_cli bm25search "science fiction"

python -m cli.semantic_search_cli search "alien invasion"
python -m cli.hybrid_search_cli search "space opera"

python -m cli.multimodal_search_cli image_search path/to/query.jpg

python -m cli.augmented_generation_cli rag "who directed The Matrix?"
python -m cli.augmented_generation_cli summarize "best romantic comedies"
```

Each script implements `--help` output with detailed command‑specific
descriptions; you can also examine the source files under `cli/`.

### Augmented Generation

The augmenters pull the top‑k documents from the hybrid retriever and then
dispatch a templated prompt to the GenAI API.  You can customize the
`TEMPLATES` dictionary in `augmented_generation_cli.py` to suit your own
application.

### Evaluation

`evaluation_cli.py` compares the engine’s ranked lists to a small “golden”
dataset, computing metrics such as precision, recall, and MRR.  Run it with:

```bash
python -m cli.evaluation_cli --help
``` 

## 🧩 Extending the Project

- **Swap the corpus.** Update `PATH_MOVIES_FILE` in `cli/settings.py` and
  regenerate any caches (`python -m cli.semantic_search_cli` etc.).
- **Use different models.** The constructors of `SemanticSearch`,
  `HybridSearch`, and `MultimodalSearch` accept a model name string.
- **Add new search strategies.** All of the retrieval logic lives under
  `cli/lib`; you can write new classes and expose them through new CLI tools.

## ⚡ Dependencies

The project relies on:

- `sentence-transformers` for embedding text and images
- `torch` / `timm` for model runtimes
- `numpy`, `Pillow`, `nltk` for data handling
- `google-genai` for calling Google’s GenAI models

See `pyproject.toml` for the exact versions used during development.

## 🙋‍♂️ Author & License

This is a sample/demo project; feel free to reuse the code under the terms of
your choice.  (No explicit license is provided.)

---

Happy experimenting with RAG! 🎉
