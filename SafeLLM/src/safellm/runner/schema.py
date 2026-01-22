# this file will create one json object per episode in a scenrio this is useful for training potential mitigation methods or detections 
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class toolcallrecord:
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
   
    