# need torch (The Engine):
# need safetensors / maybe compressed tensors (The Fuel):
# need tokenizers (The Intake):
# need transformers (The Chassis):
# need accelerate (The Transmission):
# need nnsight wrapper later (The Diagnostic Tool):
# need to work for multiple different LLMs (The Versatile Body):
from __future__ import annotations
from typing import Any, Dict
import torch
import time
from transformers import BitsAndBytesConfig
from nnterp import StandardizedTransformer

global_loaded_model = None

class HuggingFaceNNSightBackend: