from __future__ import annotations  
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from inspect_ai.tool import tool, Tool

# need a path to stop the model from over writing other files in the sandbox 
def use_path(real_path: Path, model_path: str) -> Path:
    root = real_path.resolve()
    user = Path(model_path)

    if user.is_absolute():
        raise ValueError("path are not allowed")
    path_to_use = (root / user).resolve()
    try:
        path_to_use.relative_to(root)
    except ValueError:
        raise ValueError("path outside sandbox")
    return path_to_use

def check_output(output: Path, sandbox: Path) -> None:
    output_directory = (output / "outputs").resolve()
    output = output.resolve()
    if output_directory not in output.parents and output != output_directory:
        raise ValueError ("writes need to be in outputs/.")

@tool
def read_file() -> Tool:
    """
    Read the contents of a file at the specified path.
    """
    async def excute(path: str) -> str:
        real_path = use_path(Path.cwd(), path)
        return real_path.read_text(encoding="utf-8")
    return excute

@tool # co-author: co-pilot
def list_files() -> Tool:
    """
    List all files in the directory at the specified path.
    """
    async def excute(path: str) -> List[str]:
        real_path = use_path(Path.cwd(), path)
        return [str(p.name) for p in real_path.iterdir()]
    return excute

@tool #co-author: co-pilot
def write_file() -> Tool:
    """
    Write text content to a file at the specified path.
    """
    async def excute(path: str, content: str) -> str:
        real_path = use_path(Path.cwd(), path)
        real_path.write_text(content, encoding="utf-8")
        return "File written successfully."
    return excute

@tool  # co-author: co-pilot
def make_dir() -> Tool:
    """
    Create a new directory at the specified path.
    """
    async def excute(path: str) -> str:
        real_path = use_path(Path.cwd(), path)
        real_path.mkdir(parents=True, exist_ok=True)
        return "Directory created successfully."
    return excute

@tool  # co-author: co-pilot
def append_file() -> Tool:
    """
    Append text content to the end of an existing file.
    """
    async def excute(path: str, content: str) -> str:
        real_path = use_path(Path.cwd(), path)
        with real_path.open("a", encoding="utf-8") as f:
            f.write(content)
        return "Content appended successfully."
    return excute

@tool  # co-author: co-pilot
def write_csv() -> Tool:
    """
    Write a list of dictionaries to a CSV file.
    """
    async def excute(path: str, rows: List[Dict[str, Any]]) -> str:
        real_path = use_path(Path.cwd(), path)
        if not rows:
            return "No data provided to write."
        with real_path.open("w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return "CSV file written successfully."
    return excute

@tool  # co-author: co-pilot
def read_csv() -> Tool:
    """
    Read content from a CSV file and return it as a list of dictionaries.
    """
    async def excute(path: str) -> List[Dict[str, Any]]:
        real_path = use_path(Path.cwd(), path)
        with real_path.open("r", newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
    return excute

@tool # co-author: co-pilot
def done() -> Tool:
    """
    Signal that the task is complete.
    """
    async def excute() -> str:
        return "All tasks completed."
    return excute