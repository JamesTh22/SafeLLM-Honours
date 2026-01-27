## this file is like a bridge between huggingface models and inspect ai framework
import asyncio
from typing import Any, List, Optional
from inspect_ai.model import (
    model, Model, GenerateInput, ModelOutput, ChatMessage
)
from .huggingface_nnsight import HuggingFaceNNSightBackend

@model("nnsight")
def backendNNsight(model_name: str, **model_kwargs: Any) -> Model:  # routes inspect ai to custome backend 
    return Backend(model_name, **model_kwargs)

class Backend(Model):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.backendargs = kwargs                      # store the auguments for the backend 
        self.backend = None     
    async def generate(
            self, 
        input: List[ChatMessage], 
        tools: List[Any] = [], 
        tool_choice: Any = None, 
        config: Any = None
    ) -> ModelOutput:
        
        if self.backend is None:
            self.backend = HuggingFaceNNSightBackend(self.name, **self.backend_args)  # initisalise the backend only once

          