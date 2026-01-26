# need torch (The Engine):
# need safetensors / maybe compressed tensors (The Fuel):
# need tokenizers (The Intake):
# need transformers (The Chassis):
# need accelerate (The Transmission):
# need nnsight wrapper later (The Diagnostic Tool):
# need to work for multiple different LLMs (The Versatile Body):

## HUGGINGFACE BACKEND WITH NNTERP WHICH USES NNSIGHT FOR INTERPRETABILITY THIS BACKEND ##
## IS DESIGNED TO GIVE FULL ACCESS AND CONTROL ##

from __future__ import annotations
from typing import Any, Dict
import torch
import time
from transformers import BitsAndBytesConfig
from nnterp import StandardizedTransformer

global_loaded_model = None # stop re-loading model multiple times

def model_params(model_config: Any=None) -> Dict[str, Any]:
    params = {
        "temperature": 0.7, # controlls how creative the model is (higher = more creative)
        "max_new_tokens": 2048, # if this to short model might stop mid response 
        "top_p": 0.95, # nucleus sampling do_sample must be true for this to work (controls word dictionary variety)
        "do_sample": True, # if false it will use greedy decoding mean it pick most likely next word every time
        "frequency_penalty": 0.0, # stop repeating words / control of repetition
        "presence_penalty": 0.0, # penalises a word if it has already been used like controlling topics
        "stop_strings": None, # stop model based of specific tokens
    }
    if model_config:
        if hasattr(model_config, "temperature") and model_config.temperature is not None:
            params["temperature"] = model_config.temperature
        if hasattr(model_config, "max_tokens") and model_config.max_tokens is not None:
            params["max_new_tokens"] = model_config.max_tokens
        elif hasattr(model_config, "max_new_tokens") and model_config.max_new_tokens is not None:
            params["max_new_tokens"] = model_config.max_new_tokens
        if hasattr(model_config, "top_p") and model_config.top_p is not None:
            params["top_p"] = model_config.top_p                                                           # MAPPING TO INSPACT AI PARAM NAMES
        if hasattr(model_config, "do_sample") and model_config.do_sample is not None:
            params["do_sample"] = model_config.do_sample
        if hasattr(model_config, "frequency_penalty") and model_config.frequency_penalty is not None:
            params["frequency_penalty"] = model_config.frequency_penalty
        if hasattr(model_config, "presence_penalty") and model_config.presence_penalty is not None:
            params["presence_penalty"] = model_config.presence_penalty
        if hasattr(model_config, "stop") and model_config.stop is not None:
            params["stop_strings"] = model_config.stop
        elif hasattr(model_config, "stop_strings") and model_config.stop_strings is not None:
            params["stop_strings"] = model_config.stop_strings

    if params["temperature"] == 0:
        params["do_sample"] = False

    return params

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
                bnb_4bit_compute_dtype=torch.float16
            )    
        self.model = StandardizedTransformer( # notes that is not "from pretrained" nnterp automatically handdles this
            model_name,
            quantization_config=quantization_config,
            device_map="auto", # allows accelerate to offload to CPU if GPU out of memory
            trust_remote_code=True # !!this can be a security risk (run verified models only)!!
        )
        self.tokenizer = self.model.tokenizer
        self.model_name.model = model_name
        global_loaded_model = self

    def generateloop(self, prompt: str, **kwargs) -> Dict[str, Any]:
        params = model_params(kwargs.get('config', None))
        with self.model.trace(prompt):
            hidden_states = self.model.layer_outputs[-1].outputs[0][0, -1, :].save() # last layer (-1) for the last token (-1)
            model_output = self.model.generate(
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                do_sample=params["do_sample"],
                frequency_penalty=params["frequency_penalty"],
                presence_penalty=params["presence_penalty"],
                stop_strings=params["stop_strings"], 
                pad_token_id=self.tokenizer.eos_token_id # Prevents warnings
            )
        convert_tokens_back = self.tokenizer.decode(model_output[0], skip_special_tokens=True)

        if convert_tokens_back.startswith(prompt):
            convert_tokens_back = convert_tokens_back[len(prompt):]
        else: 
           convert_tokens_back = convert_tokens_back
        return {
            "response": convert_tokens_back,
            "activations": hidden_states.value.cpu().clone()
        }
    