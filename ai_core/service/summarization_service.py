from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import sys
import os


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
            from util.llm.structured import StructuredLLMFactory
            self.llm = StructuredLLMFactory.create_llm( 
                llm_type = llm_type,
                api_key = llm_api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
    
    def summarize(
        self, 
        llm_model_name: str,
        instructions: str,
        output_model: Any,
        prompt_inputs: dict,
        prompt_config: dict,
        prompt_config_path: Optional[str] = None,
    ) -> dict:
        """
        Summarize the provided inputs using the configured LLM and prompt.
        """
        # Validations 
        assert set(prompt_config.keys()) == {'prompt_category', 'prompt_worker', 'prompt_type'}, "Keys do not match expected keys"
        assert 'pdf_text' in prompt_inputs, "pdf_text is required in prompt_inputs"

        response = {}
        # Process -> 
        try:
            # Fetch the prompt
            from util.prompt.get_prompt import PromptService
            prompt_service = PromptService(config_path=prompt_config_path) if prompt_config_path else PromptService()
            prompt = prompt_service.fetch(
                prompt_category = prompt_config.get("prompt_category"),
                prompt_worker = prompt_config.get("prompt_worker"),
                prompt_type = prompt_config.get("prompt_type"), 
                inputs = prompt_inputs
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
            "input_length": len(prompt_inputs.get('pdf_text').split()),
            "output_length": len(response.get('markdown_content', '')),
            "model_name": llm_model_name,
            "words_difference": len(response['markdown_content'].split()) - len(prompt_inputs.get('pdf_text').split()),
        }
        return response

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the summary.
        """
        return self.stats

    
# Example Usage 
if __name__ == "__main__":
    import os 
    from dotenv import load_dotenv
    from typing import List, Any
    load_dotenv()

    # Example usage of the OverviewSummarization class
    summarization_service = OverviewSummarization(
        llm_type="openai",
        llm_api_key=os.getenv("OPENAI_API_KEY")     
    )
    print(summarization_service) 

    print("---"*18)

    from pydantic import BaseModel
    class SummaryOutputModel(BaseModel):
        markdown_content: str 
        followup_questions: List[str]

    sample_pdf_text = """
        Antonio Narciso Luna de San Pedro y Novicio Ancheta (Spanish: [anˈtonjo ˈluna]; October 29, 1866 – June 5, 1899) was a Filipino army general and a pharmacist who fought in the Philippine–American War before his assassination on June 5, 1899, at the age of 32.[1][2]

        Regarded as one of the fiercest generals of his time, he succeeded Artemio Ricarte as the Commanding General of the Philippine Army. He sought to apply his background in military science to the fledgling army. A sharpshooter himself, 
        he organized professional guerrilla soldiers later named the "Luna Sharpshooters," and the "Black Guard" with Michael Joaquin. His three-tier defense, now known as the Luna Defense Line, gave the American troops a difficult endeavor during their campaign in the provinces north of Manila. 
        This defense line culminated in the creation of a military stronghold in the Cordillera.[3][page needed]

        Despite his commitment to discipline the army and serve the Republic which attracted the admiration of the people, his temper and fiery outlashes caused some to abhor him, including people from Aguinaldo's cabinet.[4] Nevertheless, Luna's efforts were recognized during his time, and he was awarded the Philippine Republic Medal in 1899. He was also a member of the Malolos Congress.[5] Besides his military studies, Luna also studied pharmacology, literature, and chemistry.[6] 
    """
    s = summarization_service.summarize(
        llm_model_name="gpt-4.1-nano",
        instructions="You are a summarization expert.",
        output_model=SummaryOutputModel,
        prompt_inputs={
            "pdf_text": sample_pdf_text, 
            "num_core_points": 3, 
            "num_detailed_points": 3, 
            "num_followup_questions": 5
        },
        prompt_config={
            "prompt_category": "overview",
            "prompt_worker": "summarizer",
            "prompt_type": "short"
        },
        prompt_config_path=None
    )
    print(s)
        



    



