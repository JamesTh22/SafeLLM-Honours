from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import EpisodeTrace


def _default_run_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("data") / "runs" / "raw" / f"runs_{ts}.jsonl"

