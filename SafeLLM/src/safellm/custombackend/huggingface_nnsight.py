# need torch (The Engine):
# need safetensors / maybe compressed tensors (The Fuel):
# need tokenizers (The Intake):
# need transformers (The Chassis):
# need accelerate (The Transmission):
# need nnsight wrapper later (The Diagnostic Tool):
# need to work for multiple different LLMs (The Versatile Body):
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig
from nnterp import StandardizedTransformer


