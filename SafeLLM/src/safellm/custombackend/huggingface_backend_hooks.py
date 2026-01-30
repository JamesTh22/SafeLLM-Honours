## HUGGINGFACE BACKEND##
## IS DESIGNED TO GIVE FULL ACCESS AND CONTROL ##

from __future__ import annotations
from typing import Any, Dict
import torch
import time
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

global_loaded_model = None # stop re-loading model multiple times

def model_params(model_config: Any=None) -> Dict[str, Any]:
    params = {
        "temperature": 0.7, # controlls how creative the model is (higher = more creative)
        "max_new_tokens": 2048, # if this to short model might stop mid response 
        "top_p": 0.95, # nucleus sampling do_sample must be true for this to work (controls word dictionary variety)
        "do_sample": True, # if false it will use greedy decoding mean it pick most likely next word every time
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

    if params["temperature"] == 0:
        params["do_sample"] = False

    return params

class HuggingFaceBackend:
    def __init__(self, model_name: str, quantisation_4bit: bool = True):
        
        global global_loaded_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

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
        self.model = AutoModelForCausalLM.from_pretrained( # pre-trained model from huggingface hub
            model_name,
            quantization_config=quantization_config,
            device_map="auto", # allows accelerate to offload to CPU if GPU out of memory
            trust_remote_code=True # !!this can be a security risk (run verified models only)!!
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        global_loaded_model = self

    def huggingface_generate_loop() # generates response from the model

    def get_model_layers() # identifes the layer of the model GPT-OSS(20b,120b),Gemma(4B,12B,27B),Llama(3B,7B,13B,70B),Mistral(14B,8B,3B) deepseek() qwen()

    def get_model_activations() # gets activations from the model uses the get_model_layers() function

    def get_logits() # gets logits from the model

    def create_activations_hooks() # creates activations hooks for the model

    def create_logits_hooks() # creates logits hooks for the model
        


