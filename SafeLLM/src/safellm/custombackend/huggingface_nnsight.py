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
from contextlib import nullcontext
import torch
import gc
import time
from transformers import BitsAndBytesConfig
from nnsight import LanguageModel

global_loaded_model = None # stop re-loading model multiple times

def model_params(model_config: Any=None) -> Dict[str, Any]:
    params = {
        "temperature": 0.7, # controlls how creative the model is (higher = more creative)
        "max_new_tokens": 128, # if this to short model might stop mid response 
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

class HuggingFaceNNSightBackend:
    def __init__(self, model_name: str, quantisation_4bit: bool = True):
        global global_loaded_model
        # Add imports if not present at module level, but they are.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.quantisation_4bit = quantisation_4bit

        # Clear cache before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if global_loaded_model is not None and global_loaded_model.model_name == model_name:
            cached_ok = True
            if getattr(global_loaded_model, "device", None) != self.device:
                cached_ok = False
            if getattr(global_loaded_model, "quantisation_4bit", None) != quantisation_4bit:
                cached_ok = False
            # Check if model is properly loaded (not on meta)
            if cached_ok:
                try:
                     # Access inner model if wrapped
                    inner_model = global_loaded_model.model._model if hasattr(global_loaded_model.model, '_model') else global_loaded_model.model
                    param_device = next(inner_model.parameters()).device.type
                    if param_device == "meta":
                        cached_ok = False
                except Exception:
                    pass
            
            if cached_ok:
                self.model = global_loaded_model.model
                self.tokenizer = global_loaded_model.tokenizer
                return

        quantization_config = None
        if quantisation_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )    
        
        if torch.cuda.is_available():
            device_map = "auto" 
        else:
            device_map = None
            
        # Manually load model to ensure it respects 4-bit and device map properly
        # preventing double loading or meta-device issues with nnsight
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Wrap in nnsight
        self.model = LanguageModel(hf_model, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        global_loaded_model = self

    def generateloop(self, prompt: str, **kwargs) -> Dict[str, Any]:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        params = model_params(kwargs.get('config', None))
        tokenized_input = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized_input.input_ids.to(self.device)
        attention_mask = tokenized_input.attention_mask.to(self.device) if "attention_mask" in tokenized_input else None
        
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            modeloutputids = self.model._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                do_sample=params["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id, # Prevents warnings
                repetition_penalty=1.2, # Prevents system prompt hallucination loops
            )
        
        # Slice generated tokens to avoid returning prompt in completion
        # logic: modeloutputids contains [input_ids + generated_ids]
        input_length = input_ids.shape[1]
        generated_ids = modeloutputids[0][input_length:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # FIX: The model sometimes hallucinates the System Prompt or User Prompt loop at the end.
        # We must strip this to prevent context explosion.
        # Check for common hallucination tokens like "system" or "user" at start of line
        hallucination_triggers = ["system\n", "user\n", "<|im_start|>", "System:"]
        for trigger in hallucination_triggers:
            if trigger in completion:
                stop_index = completion.find(trigger)
                if stop_index >= 0:
                     completion = completion[:stop_index].strip()
        
        convert_tokens_back = self.tokenizer.decode(modeloutputids[0], skip_special_tokens=True)

        # Clean memory between passes
        del input_ids, modeloutputids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set environment variable to help with fragmentation
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        hidden_states_proxy = None
        # Ensure we are in inference mode for tracing to avoid gradient overhead
        with torch.inference_mode():
            with self.model.trace(convert_tokens_back, remote=False) as tracer:
                # Save the activations first
                hidden_states_proxy = self.model.model.layers[-1].output[0].save() # last layer (-1) for the last token (-1)
                # Stop the lm_head from running to save massive memory allocation (logits for all tokens)
                # Stop the execution after this point (skipping lm_head)
                # This prevents OOM by avoiding the massive logits calculation
                if hasattr(tracer, "stop"):
                    tracer.stop()
                else:
                     if hasattr(self.model, "lm_head"):
                        self.model.lm_head.stop() 

            try:
                if hidden_states_proxy is not None:
                    tensor_shape = hidden_states_proxy.value
                else:
                    # Should not happen if trace succeeds, but handling safety
                    tensor_shape = torch.tensor([]) 
            except AttributeError:
                tensor_shape = hidden_states_proxy

        if len(tensor_shape.shape) == 3:  # If it's 3D take the last token.
            final_activation = tensor_shape[:, -1, :]
        elif len(tensor_shape.shape) == 2:
            final_activation = tensor_shape[-1, :]
        else:
            final_activation = tensor_shape  # Unexpected shape

        if convert_tokens_back.startswith(prompt):
            completion = convert_tokens_back[len(prompt):]
        else: 
            completion = convert_tokens_back
        return {
            "completion": completion,
            "activation": final_activation.detach().cpu().clone()  # put the calculations on the cpu to save gpu memory 
        }
