from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm.structured import StructuredLLMFactory 


# -- Base Class --
class SummarizationServiceBase(ABC):
    """
    Abstract base class for summarization services.
    """
    @abstractmethod 
    def summarize(self) -> Dict[str, Any]:
        """
        Summarize the provided inputs using the configured LLM and prompt.
        """
        pass 


# -- Concrete Implementation --
class OverviewSummarization(SummarizationServiceBase):
    """
    Class for summarization service of Overview Features.
    # NOTE: Works on full pdf text only.

    - Short summary 
    - Detailed summary
    """
    def __init__(self, llm_type: str, llm_api_key: str) -> None: 
        self.stats = {}
        # init llm
        try:
            self.llm = StructuredLLMFactory.create_llm( 
                llm_type = llm_type,
                api_key = llm_api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
    
    def summarize(
        self, 
        pdf_text: str,
        llm_model_name: str,
        instructions: str,
        output_model: Any,
        prompt_inputs: dict,
        prompt_config_path: Optional[str] = None,
    ) -> str:
        """
        Summarize the provided inputs using the configured LLM and prompt.
        """
        # Validations 
        assert set(prompt_inputs.keys()) == {'prompt_category', 'prompt_worker', 'prompt_type'}, "Keys do not match expected keys"
        
        response = {}
        # Process -> 
        try:
            # Fetch the prompt
            from prompt.get_prompt import PromptService
            prompt_service = PromptService(config_path=prompt_config_path) if prompt_config_path else PromptService()
            prompt = prompt_service.fetch(
                prompt_category = prompt_inputs.get("prompt_category"),
                prompt_worker = prompt_inputs.get("prompt_worker"),
                prompt_type = prompt_inputs.get("prompt_type"), 
                pdf_text = pdf_text
            )
            assert prompt, "Prompt cannot be empty. Please check the prompt configuration."
        except Exception as e:
            raise ValueError(f"Failed to fetch prompt: {e}")
        
        try:
            # Generate summary using the LLM
            response = self.llm.generate(
                message = prompt,
                model_name = llm_model_name,
                output_model = output_model,
                instructions = instructions
            )
            assert response, "LLM response cannot be empty. Please check the LLM configuration."
        except Exception as e:
            raise ValueError(f"Failed to generate summary: {e}")
        
        # Store stats
        self.stats = {
            "input_length": len(pdf_text),
            "output_length": len(response.get('summary', '')),
            "model_name": llm_model_name,
            "words_difference": len(response['summary'].split()) - len(pdf_text.split()),
        }
        return response

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the summary.
        """
        return self.stats

    
# Example Usage 
if __name__ == "__main__":
    # Example usage of the OverviewSummarization class
    summarization_service = OverviewSummarization(
        llm_type="openai",
        llm_api_key=os.getenv("OPENAI_API_KEY")     
    )
    print(summarization_service) 



    



