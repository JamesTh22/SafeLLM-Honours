## this file is like a bridge between huggingface models and inspect ai framework
import asyncio
from typing import Any, List, Optional
from inspect_ai.model import (
    ModelAPI, modelapi, ModelOutput, ChatMessage
)
from .huggingface_nnsight import HuggingFaceNNSightBackend

@modelapi(name="nnsight") # routes inspect ai to custome backend 
class Backend(ModelAPI):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name)
        self.backend_args = kwargs                      # store the auguments for the backend 
        self.backend = None     
    async def generate(
        self, 
        input: List[ChatMessage], 
        tools: List[Any] = [], 
        tool_choice: Any = None, 
        config: Any = None
    ) -> ModelOutput:
        
        if self.backend is None:
            copyargs = self.backend_args.copy()
            cleanargs = copyargs.copy()
            cleanargs.pop("base_url", None) 
            cleanargs.pop("api_key", None) # remove api_key from inspect ai
            cleanargs.pop("config", None) # remove config from inspect ai
            cleanargs.pop("max_tokens", None)


            self.backend = HuggingFaceNNSightBackend(self.model_name, **cleanargs)  # initisalise the backend only once
        model_chat_message = [{"role": msg.role, "content": msg.content} for msg in input]
        prompt = self.backend.tokenizer.apply_chat_template(            # coverts string in to a chat format for the model to understand
            model_chat_message,                                         # uses the offical apply chat template this normlises stuff for different models
            tokenize=False,                                             # since every model has it own config.json which is different       
            add_generation_prompt=True
        )
        def run_on_gpu():
            return self.backend.generateloop(prompt, config=config)
        result = await asyncio.to_thread(run_on_gpu)  # run the blocking gpu code in a thread

        return ModelOutput(
            model=self.model_name,
            completion=result["completion"],
            metadata={
                "activation": result["activation"]
            }
        )