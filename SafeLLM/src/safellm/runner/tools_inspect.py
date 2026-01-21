from __future__ import annotations  
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from inspect_ai.tool import tool, TOOL

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