import os
import subprocess
import sys
from pathlib import Path

PIPELINES = {
    "hourly": ["data_extraction.py", "feature_engineering.py"],
    "daily": ["model_train.py", "promote_model.py"],
}


def get_pipeline_steps(pipeline_name):
    try:
        return PIPELINES[pipeline_name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(PIPELINES.keys()))
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Expected one of: {valid}") from exc


def _project_root():
    return Path(__file__).resolve().parent.parent


def run_pipeline(pipeline_name):
    project_root = _project_root()
    scripts_dir = project_root / "scripts"

    for script_name in get_pipeline_steps(pipeline_name):
        target = scripts_dir / script_name
        print(f"\n--- Running: {script_name} ---")
        result = subprocess.run([sys.executable, str(target)], cwd=str(project_root), capture_output=False)
        if result.returncode != 0:
            print(f"FAILED: {script_name} exited with error.")
            raise SystemExit(result.returncode)

    print(f"\n[SUCCESS] {pipeline_name.title()} pipeline finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a project data or model pipeline.")
    parser.add_argument("pipeline", choices=sorted(PIPELINES.keys()), help="Pipeline name")
    args = parser.parse_args()
    run_pipeline(args.pipeline)
