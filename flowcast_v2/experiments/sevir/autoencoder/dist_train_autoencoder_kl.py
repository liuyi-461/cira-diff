"""Compatibility entrypoint; use experiments.goes for GOES training."""
import sys
from pathlib import Path
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if __name__ == "__main__":
    runpy.run_module("experiments.goes.autoencoder.dist_train_autoencoder_kl", run_name="__main__")
