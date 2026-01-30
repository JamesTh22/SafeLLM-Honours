# gets gets jsonl data using the blueprint file schema.py 
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import EpisodeTrace


def _default_run_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("data") / "runs" / "raw" / f"runs_{ts}.jsonl"


def append_jsonl(record: Dict[str, Any], path: Optional[Path] = None) -> Path:
    path = path or _default_run_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_episode_trace(trace: EpisodeTrace, path: Optional[Path] = None) -> Path:
    return append_jsonl(trace.to_dict(), path=path)