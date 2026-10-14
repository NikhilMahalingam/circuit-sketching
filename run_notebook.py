"""Execute the research notebook in the project's registered CUDA kernel.

Run with .\.venv\Scripts\python.exe run_notebook.py after installing requirements
and registering the mib-sketching kernel. Keeps the source notebook clean and
saves actual outputs to mib_sketching_baseline.executed.ipynb.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
for variable, relative in {
    "HF_HOME": ".cache/huggingface",
    "MPLCONFIGDIR": ".cache/matplotlib",
    "IPYTHONDIR": ".cache/ipython",
    "JUPYTER_RUNTIME_DIR": ".cache/jupyter",
}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import nbformat
from nbclient import NotebookClient
import torch


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        first_line = next((line for line in cell.source.splitlines() if line.strip()), "")
        print(f"Executing cell {cell_index + 1}: {first_line[:100]}", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in {sys.executable}; select the project .venv.")
    print("Python:", sys.executable, flush=True)
    print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0), flush=True)
    source = ROOT / "mib_sketching_baseline.ipynb"
    output = ROOT / "mib_sketching_baseline.executed.ipynb"
    results_root = ROOT / "results"
    previous_results = set(results_root.iterdir()) if results_root.exists() else set()
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    print("Notebook validated; preparing the Jupyter kernel.", flush=True)
    client = NotebookClient(
        notebook, timeout=1200, kernel_name="mib-sketching", allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}}, on_cell_execute=progress,
    )
    started = time.perf_counter()
    try:
        print("Starting kernel and executing cells.", flush=True)
        client.execute()
    except Exception:
        failure_path = ROOT / "review_checks" / "notebook_execution.failed.ipynb"
        nbformat.write(notebook, failure_path)
        print("Partial execution saved to:", failure_path, flush=True)
        raise
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(out.output_type == "error" for cell in code_cells for out in cell.outputs)
    new_results = set(results_root.iterdir()) - previous_results
    assert len(new_results) == 1, "Expected one newly exported experiment directory."
    results_directory = new_results.pop()
    configuration = json.loads((results_directory / "configuration.json").read_text(encoding="utf-8"))
    assert configuration["device"] == "cuda", "Notebook kernel did not use CUDA."
    nbformat.validate(notebook)
    nbformat.write(notebook, output)
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "executed_code_cells": len(code_cells),
        "wall_seconds_including_kernel_and_downloads": time.perf_counter() - started,
        "output_notebook": str(output), "results_directory": str(results_directory),
    }
    report_path = ROOT / "review_checks" / "cuda_execution.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
