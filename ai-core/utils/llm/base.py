from abc import ABC, abstractmethod
import os, sys, yaml
from typing import Any, Dict, Optional


class ConfigManager:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'llm_config.yaml')
        try:
            with open(config_path, 'r') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using empty configuration.")
            self._config = {"llm_configs": {}}
    
    def get_config(self, llm_type, mode):
        """Get configuration for a specific LLM type and mode"""
        return self._config.get("llm_configs", {}).get(llm_type, {}).get(mode, {})

    
class LLMBase(ABC): 
    def __init__(
        self, 
        api_key: str, 
        model_name: Optional[str] = None,
        **kwargs
    ) -> None: 
        assert api_key and isinstance(api_key, str), "API key invalid or missing."
        
        # Load default config for this LLM type and mode
        self.config_manager = ConfigManager()
        self.llm_type = self._get_llm_type()
        self.mode = self._get_mode()
        self.default_config = self.config_manager.get_config(self.llm_type, self.mode)
        
        # Use provided model_name or fall back to default
        self.model_name = model_name if model_name else self.default_config.get("default_model")
        assert self.model_name and isinstance(self.model_name, str), "Model name invalid or missing."
        
        self.api_key = api_key
        
        # Store generation parameters (combining defaults with user overrides)
        self.generation_params = {**self.default_config}
        # Remove default_model from generation params
        if "default_model" in self.generation_params:
            del self.generation_params["default_model"]
        # Update with any user-provided parameters
        self.generation_params.update(kwargs)
    
    @abstractmethod
    def _get_llm_type(self) -> str:
        """Return the type of LLM (e.g., 'openai', 'anthropic', etc.)"""
        pass
    
    @abstractmethod
    def _get_mode(self) -> str:
        """Return the mode (general or structured)"""
        pass
    
    def update_model(self, model_name: str) -> None: 
        """Update the model name."""
        self.model_name = model_name
    
    @abstractmethod
    def list_models(self) -> list:
        """List available models."""
        pass

    @abstractmethod
    def generate(self, prompt: str, response_cls: Optional[Any] = None, **override_params) -> Any:
        """Generate text based on the provided prompt."""
        pass