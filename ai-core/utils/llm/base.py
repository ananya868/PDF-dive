from abc import ABC, abstractmethod
import os, sys, yaml
from typing import Any, Dict, Optional


# -- LLM Base class -- 
class LLMBase(ABC):
    """Base class for all LLMs. Provides common functionality and configuration management."""
    def __init__(
        self,
        api_key: str
    ) -> None: 
        """
        Initialize the LLM with an API key, optional model name, and additional parameters.
            :param api_key: The API key for accessing the LLM service.
            :param model_name: Optional name of the model to use. If not provided, defaults to the configured model.
            :param kwargs: Additional parameters for generation, which will override default settings.
        """
        assert api_key and isinstance(api_key, str), "API key invalid or missing."

        # Load default config for this LLM type and mode
        self.llm_type = self._get_llm_type()
        self.mode = self._get_mode()
        
        # Load configuration from YAML file
        config_path = os.path.join(os.path.dirname(__file__), 'llm_config.yaml')
        # Open yaml
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)         
        except Exception as e: 
            print(f"Config file error: {e}. Using empty configuration.")
            self.config = {}
        self.api_key = api_key 

    @abstractmethod
    def _get_mode(self) -> str:
        """Return the mode (general or structured)"""
        pass

    @abstractmethod
    def list_models(self) -> list:
        """List available models."""
        pass

    @abstractmethod
    def generate(self, message: str, model_name: str, instructions: str) -> Any:
        """Generate text based on the provided prompt."""
        pass

