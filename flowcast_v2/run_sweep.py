"""Compatibility entrypoint for python -m tools.run_sweep."""
from tools.run_sweep import main
if __name__ == '__main__':
    raise SystemExit(main())
