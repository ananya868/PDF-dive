from typing import Any, List, Dict, Optional
import sys, os 


class ExplainService:
    def __init__(
        self,
        llm_type: str,
        llm_api_key: str
    ) -> None:
        # Initialize client
        try:
            from util.llm.general import GeneralLLMFactory
            self.llm = GeneralLLMFactory.create_llm(
                llm_type=llm_type,
                api_key=llm_api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
    
    def explain(
        self,
        llm_model_name: str,
        instructions: str,
        prompt_inputs: Dict[str, Any],
        prompt_config: Dict[str, Any] = {
            'prompt_category': 'quick_access',
            'prompt_worker': 'explainer',
            'prompt_type': 'quick'
        }
    ) -> dict:
        """
        Explain the provided inputs using the configured LLM and prompt.
        """
        try:
            # Fetch the prompt 
            from util.prompt.get_prompt import PromptService
            prompt = PromptService().fetch(
                prompt_category=prompt_config['prompt_category'],
                prompt_worker=prompt_config['prompt_worker'],
                prompt_type=prompt_config['prompt_type'],
                inputs=prompt_inputs
            )
            assert prompt, "Prompt not found for the given configuration"
        except Exception as e:
            raise ValueError(f"Failed to fetch prompt: {e}")
        
        try:
            # Generate explanation using the LLM
            generated_text = self.llm.generate(
                message=prompt,
                model_name=llm_model_name,
                instructions=instructions
            )
            assert generated_text, "LLM response cannot be empty. Please check the LLM configuration."
        except Exception as e:
            raise ValueError(f"Failed to generate explanation: {e}")
        response = {
            'explanation': generated_text,
        }
        return response

        
# usage 
if __name__ == "__main__":
    import os 
    from dotenv import load_dotenv
    load_dotenv()
    # Example usage
    explain_service = ExplainService(llm_type='openai', llm_api_key=os.getenv('OPENAI_API_KEY'))

    selected_text = """
    The quick brown fox jumps over the lazy dog. This is a classic example of a pangram, which is a sentence that contains every letter of the alphabet at least once. Pangrams are often used in typography and testing fonts because they showcase the full range of letters in a concise manner.
    """

    instructions = "Provide a brief explanation of the selected text."
    prompt_inputs = {
        'selected_text': selected_text,
        'max_words': 50
    }
    response = explain_service.explain(
        llm_model_name="gpt-4.1-nano",
        instructions=instructions,
        prompt_inputs=prompt_inputs
    )
    print(response)