from typing import Any, Optional
from base import LLMBase


# -- Mode --
class GeneralLLMBase(LLMBase):
    def _get_mode(self) -> str:
        return "general"
    

# -- LLM Classes --
class OpenAIGeneral(GeneralLLMBase):
    def __init__(self, api_key: str) -> Any:
        super().__init__(api_key) 
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def list_models(self) -> list:
        models = self.config.get("llm_general").get("openai").get("models")
        return models
    
    def generate(self, message: str, model_name: str, instructions: str, **kwargs) -> Any:
        response_text = ""
        try:
            response = self.client.responses.create(
                model = model_name, 
                instructions = instructions,
                input = message
            )
            response_text = response.output_text
        except Exception as e:
            response_text = f"Some error occurred: {str(e)}"
        return response_text


class GoogleAIGeneral(GeneralLLMBase):
    def __init__(self, api_key: str) -> Any: 
        super().__init__(api_key) 
        
        from google import genai
        from google.genai.types import HttpOptions
        self.client = genai.Client(https_options = HttpOptions(api_key = api_key, api_version="v1"))
    
    def list_models(self) -> list:
        models = self.config.get("llm_general").get("google").get("models")
        return models
    
    def generate(self, message: str, model_name: str, instructions: str, **kwargs) -> Any:
        response_text = ""
        try: 
            response = self.client.models.generate_text(
                model = model_name,
                contents = message
            )
            response_text = response.text
        except Exception as e:
            response_text = f"Some error occurred: {str(e)}"
        return response_text


class GroqAIGeneral(GeneralLLMBase):
    def __init__(self, api_key: str) -> Any: 
        super().__init__(api_key) 
        
        from groq import Groq
        self.client = Groq(api_key=api_key)

    def list_models(self) -> list:
        models = self.config.get("llm_general").get("groq").get("models")
        return models
    
    def generate(self, message: str, model_name: str, instructions: str, **kwargs) -> Any:
        response_text = ""
        try: 
            response = self.client.chat.completions.create(
                messages = [
                    {
                        "role": "system",
                        "content": instructions 
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model = model_name
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            response_text = f"Some error occurred: {str(e)}"
        return response_text


class MistralAIGeneral(GeneralLLMBase):
    def __init__(self, api_key: str) -> Any: 
        super().__init__(api_key) 
        
        from mistralai import Mistral
        self.client = Mistral(api_key=api_key)

    def list_models(self) -> list:
        models = self.config.get("llm_general").get("mistral").get("models")
        return models
    
    def generate(self, message: str, model_name: str, instructions: str, **kwargs) -> Any:
        response_text = ""
        try: 
            response = self.client.chat.complete(
                model = model_name,
                messages = [
                    {
                        "role": "system",
                        "content": instructions
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            response_text = f"Some error occurred: {str(e)}"
        return response_text


class AnthropicAIGeneral(GeneralLLMBase):
    def __init__(self, api_key: str) -> Any: 
        super().__init__(api_key) 
        
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def list_models(self) -> list:
        models = self.config.get("llm_general").get("anthropic").get("models")
        return models

    def generate(self, message: str, model_name: str, instructions: str, **kwargs) -> Any:
        response_text = ""
        try: 
            response = self.client.messages.create(
                model = model_name,
                temperature = 0.7,
                system = instructions,
                messages = [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            )
            response_text = response.content[0].text
        except Exception as e:
            response_text = f"Some error occurred: {str(e)}"
        return response_text


# -- Factory Class --
class GeneralLLMFactory:
    @classmethod
    def list_llm(cls) -> dict:
        import os, yaml 

        llm_list = None
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'llm_config.yaml')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            llm_list = config.get("llm_general")
        except Exception as e:
            llm_list = {}
        return llm_list
    
    @staticmethod 
    def create_llm(llm_type: str, api_key: str) -> Any:
        llm_classes = {
            "openai": OpenAIGeneral,
            "google": GoogleAIGeneral,
            "groq": GroqAIGeneral,
            "mistral": MistralAIGeneral,
            "anthropic": AnthropicAIGeneral
        }
        
        llm_class = llm_classes.get(llm_type.lower())
        if llm_class:
            return llm_class(api_key)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")


# Example usage:
if __name__ == "__main__":
    llm_factory = GeneralLLMFactory()
    llm = llm_factory.create_llm("anthropic", "your_api_key_here")
    print(llm.list_models())
    response = llm.generate("Hello, how are you?", "gpt-3.5-turbo", "Respond in a friendly manner.")
    print(response)
    llm = llm_factory.create_llm("google", "your_api_key_here")
    print(llm.list_models())
    response = llm.generate("Hello, how are you?", "gemini-1.5-pro", "Respond in a friendly manner.")
    print(response)
