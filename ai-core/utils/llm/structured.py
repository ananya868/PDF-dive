from typing import Any, Optional, Dict
from .base import LLMBase


class StructuredLLMBase(LLMBase):
    def _get_mode(self) -> str:
        return "structured"


class OpenAIStructured(StructuredLLMBase):
    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs) -> None:
        super().__init__(api_key, model_name, **kwargs)
        # -- Client -- 
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def _get_llm_type(self) -> str:
        return "openai"

    def list_models(self) -> list:
        models = self.client.models.list()
        return [model.id for model in models.data]

    def generate(self, prompt: str, response_format: Optional[Dict] = None, response_cls: Optional[Any] = None, **override_params) -> Any:
        # Combine default generation parameters with any overrides for this specific call
        params = {**self.generation_params, **override_params}
        
        # Set up JSON mode if no specific response format is provided
        if response_format is None:
            response_format = {"type": "json_object"}
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            **params
        )
        
        if response_cls:
            # Assume response_cls knows how to parse the response
            return response_cls(response)
        return response.choices[0].message.content


class AnthropicStructured(StructuredLLMBase):
    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs) -> None:
        super().__init__(api_key, model_name, **kwargs)
        # -- Client -- 
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)

    def _get_llm_type(self) -> str:
        return "anthropic"

    def list_models(self) -> list:
        # Note: Anthropic doesn't have a list_models API as of now
        return ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

    def generate(self, prompt: str, response_cls: Optional[Any] = None, **override_params) -> Any:
        # Combine default generation parameters with any overrides
        params = {**self.generation_params, **override_params}
        
        # Add system prompt for structured output
        system_prompt = "Return your response as a JSON object with appropriate keys and values."
        
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        
        if response_cls:
            return response_cls(response)
        return response.content[0].text


class MistralStructured(StructuredLLMBase):
    def __init__(self, api_key: str, model_name: Optional[str] = None, **kwargs) -> None:
        super().__init__(api_key, model_name, **kwargs)
        # -- Client -- 
        from mistralai.client import MistralClient
        self.client = MistralClient(api_key=self.api_key)

    def _get_llm_type(self) -> str:
        return "mistral"

    def list_models(self) -> list:
        return ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"]

    def generate(self, prompt: str, response_cls: Optional[Any] = None, **override_params) -> Any:
        # Combine default generation parameters with any overrides
        params = {**self.generation_params, **override_params}
        
        # Add system prompt for structured output
        system_prompt = "Return your response as a JSON object with appropriate keys and values."
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **params
        )
        
        if response_cls:
            return response_cls(response)
        return response.choices[0].message.content