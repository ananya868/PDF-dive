from typing import Any, Optional
from base import LLMBase


# -- Mode --
class StructuredLLMBase(LLMBase):
    def _get_mode(self) -> str:
        return "structured"
    

# -- LLM Classes --
class OpenAIStructured(StructuredLLMBase):
    def __init__(self, api_key: str) -> None:
        super().__init__(api_key) 
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def list_models(self) -> list:
        models = self.config.get("llm_structured").get("openai").get("models")
        return models
    
    def generate(self, message: str, model_name: str, output_model: Any, instructions: str, **kwargs) -> Any:
        response_cls = None
        try:
            response = self.client.responses.parse(
                model = model_name,
                instructions = instructions,
                input = message,
                text_format = output_model
            )
            event = response.output_parsed 
            response_cls = event.model_dump()
        except Exception as e:
            response_cls = {}
        return response_cls


class GoogleAIStructured(StructuredLLMBase):
    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        
        from google import genai 
        self.client = genai.Client(api_key=self.api_key)
    
    def list_models(self) -> list:
        models = self.config.get("llm_structured").get("google").get("models")
        return models

    def generate(self, message: str, model_name: str, output_model: Any, instructions: str, **kwargs) -> Any:
        response_cls = None
        try:
            response = self.client.models.generate_content(
                model = model_name, 
                contents = message, 
                config = {
                    "response_mime_type": "application/json",
                    "response_schema": output_model
                }
            )
            event = response.parsed 
            response_cls = event[0].model_dump()
        except Exception as e:
            response_cls = {}
        return response_cls


class GroqAIStructured(StructuredLLMBase):
    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)

        from groq import Groq
        self.client = Groq(api_key=api_key)

    def list_models(self) -> list:
        models = self.config.get("llm_structured").get("groq").get("models")
        return models

    def generate(self, message: str, model_name: str, output_model: Any, instructions: str, **kwargs) -> Any:
        response_cls = None
        try:
            response = self.client.chat.completions.create(
                messages = [
                    {
                        "role": "system",
                        "content": "{instructions}\n".format(instructions=instructions) +
                        f" The JSON object must use the schema: {json.dumps(output_model.model_json_schema(), indent=2)}"
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model = model_name,
                temperature = 0,
                stream = False
            )
            event = output_model.model_validate_json(response.choices[0].message.content)
            response_cls = event.model_dump()
        except Exception as e:
            response_cls = {}
        return response_cls


class MistralAIStructured(StructuredLLMBase):
    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)

        from mistralai import Mistral
        self.client = Mistral(api_key=api_key)

    def list_models(self) -> list:  
        models = self.config.get("llm_structured").get("mistral").get("models")
        return models
    
    def generate(self, message: str, model_name: str, output_model: Any, instructions: str, **kwargs) -> Any:
        response_cls = None
        try:
            response = self.client.chat.parse(
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
                ], 
                response_format = output_model,
                max_tokens = 3000, 
                temperature = 0
            )
            event = response.choices[0].message.content
            response_cls = event 
        except Exception as e:
            response_cls = {}
        return response_cls 
    

# -- Factory Class -- 
class StructuredLLMFactory:
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
        """Factory method to create an instance of a structured LLM class based on the type."""
        llm_classes = {
            "openai": OpenAIStructured,
            "google": GoogleAIStructured,
            "groq": GroqAIStructured,
            "mistral": MistralAIStructured
        }
        return llm_classes.get(llm_type.lower())(api_key) if llm_type.lower() in llm_classes else None
    