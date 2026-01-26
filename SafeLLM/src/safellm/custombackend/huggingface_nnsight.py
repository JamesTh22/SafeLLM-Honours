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

global_loaded_model = None # stop re-loading model multiple times

class HuggingFaceNNSightBackend:
    def __init__(self, model_name: str, quantisation_4bit: bool = True):
        global global_loaded_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if global_loaded_model is not None and global_loaded_model.model_name == model_name:
            self.model = global_loaded_model.model
            self.tokenizer = global_loaded_model.tokenizer
            return
        quantization_config = None
        if quantisation_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",    #nf4 is slightly better than fp4 for accuracy / this save more memory 
                bnb_4bit_compute_type=torch.float16
            )    