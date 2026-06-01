"""Thin shim: delegates to the packaged CLI.

Prefer `uv run -m reflectolearn` or the installed `reflectolearn` script.
This file is kept so `python main.py` still works.
"""

from reflectolearn.cli import main

if __name__ == "__main__":
    main()
