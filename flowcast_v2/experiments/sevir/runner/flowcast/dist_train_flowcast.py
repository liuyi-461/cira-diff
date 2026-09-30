"""Compatibility entrypoint; use experiments.goes for GOES training."""
import sys
from pathlib import Path
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
if __name__ == "__main__":
    runpy.run_module("experiments.goes.runner.flowcast.dist_train_flowcast", run_name="__main__")
