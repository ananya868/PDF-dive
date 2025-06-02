from typing import Any, Optional
from .base import LLMBase


class GeneralLLMBase(LLMBase):
    def _get_mode(self) -> str:
        return "general"


class OpenAIGeneral(GeneralLLMBase):
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

    def generate(self, prompt: str, response_cls: Optional[Any] = None, **override_params) -> Any:
        # Combine default generation parameters with any overrides for this specific call
        params = {**self.generation_params, **override_params}
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        
        if response_cls:
            # Assume response_cls knows how to parse the response
            return response_cls(response)
        return response.choices[0].message.content


class AnthropicGeneral(GeneralLLMBase):
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
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        
        if response_cls:
            return response_cls(response)
        return response.content[0].text


class MistralGeneral(GeneralLLMBase):
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
        
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        
        if response_cls:
            return response_cls(response)
        return response.choices[0].message.content