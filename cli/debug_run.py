import os
import shlex
import sys

from cli import keyword_search_cli


def main() -> None:
    args_str = os.environ.get("DEBUG_ARGS", "")
    argv = shlex.split(args_str) if args_str else []
    sys.argv = [keyword_search_cli.__file__] + argv
    keyword_search_cli.main()


if __name__ == "__main__":
    main()
