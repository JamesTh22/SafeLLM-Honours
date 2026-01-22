# this file will create one json object per episode in a scenrio this is useful for training potential mitigation methods or detections 
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
   
@dataclass
class MessageRecord:
    role: str
    content: str
    tool_calls: Optional[List[ToolCallRecord]] = None

@dataclass
class EpisodeRecord:
    scenario_id: str
    case_id: str
    variant: str
    model: str
    sample_id: str

    messages: List[MessageRecord]

    output: Optional[str] = None
    model_metadata: Optional[Dict[str, Any]] = None   # logprobs, hidden_states pointers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    