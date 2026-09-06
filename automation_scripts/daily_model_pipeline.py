import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.pipeline_runner import run_pipeline


if __name__ == "__main__":
    run_pipeline("daily")
