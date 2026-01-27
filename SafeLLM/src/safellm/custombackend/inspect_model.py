## this file is like a bridge between huggingface models and inspect ai framework
import asyncio
from typing import Any, List, Optional
from inspect_ai.model import (
    model, Model, GenerateInput, ModelOutput, ChatMessage
)
from .huggingface_nnsight import HuggingFaceNNSightBackend

