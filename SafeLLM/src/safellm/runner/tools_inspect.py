from __future__ import annotations  
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from inspect_ai.tool import tool, TOOL

# need a path to stop the model from over writing other files in the sandbox 
def use_path (real_path: Path, model_path: str) -> Path:
    real_path = real_path(".").resolve()

    model = Path(model_path)
    # don't accept this path 
    if model != real_path:
        raise ValueError("wrong path")
    
    path_to_use = (real_path / model).resolve()

    try:
        path_to_use.relative_to(model)
    except:
        raise ValueError("path outside sandbox")
    return path_to_use
