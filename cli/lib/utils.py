import pickle


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
