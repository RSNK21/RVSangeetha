"""PyInstaller entry point.

Must import the package absolutely: building ``rvsangeetha/__main__.py``
directly makes it a top-level script whose relative imports fail
("attempted relative import with no known parent package").
"""

from rvsangeetha.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
