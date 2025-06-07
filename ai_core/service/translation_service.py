from typing import Any, List, Dict, Optional
import sys, os 


class TranslationService: 
    def __init__(
        self,
        llm_type: str,
        llm_api_key: str
    ) -> None:
        # init client 
        try: 
            from util.llm.general import GeneralLLMFactory
            self.llm = GeneralLLMFactory.create_llm(
                llm_type=llm_type,
                api_key=llm_api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
        
    def translate(
        self,
        llm_model_name: str, 
        instructions: str, 
        prompt_inputs: Dict[str, Any],
        prompt_config: Dict[str, Any] = {
            'prompt_category': 'overview',
            'prompt_worker': 'translator',
            'prompt_type': 'detailed'
        }
    ) -> dict:
        """
        Translate the provided inputs using the configured LLM and prompt.
        """ 
        # Validations 
        assert 'pdf_text' in prompt_inputs, "pdf_text is required in prompt_inputs"

        response = {}
        # Process ->
        try:
            # Fetch the prompt 
            from util.prompt.get_prompt import PromptService
            translation_prompt = PromptService().fetch(
                prompt_category=prompt_config['prompt_category'],
                prompt_worker=prompt_config['prompt_worker'],
                prompt_type=prompt_config['prompt_type'],
                inputs = prompt_inputs
            )
            assert translation_prompt, "Prompt not found for the given configuration"
        except Exception as e:
            raise ValueError(f"Failed to fetch prompt: {e}")
        
        try: 
            # Generate translation using the LLM
            generated_text = self.llm.generate(
                message = translation_prompt,
                model_name = llm_model_name,
                instructions = instructions
            )
            assert generated_text, "LLM response cannot be empty. Please check the LLM configuration."
        except Exception as e:
            raise ValueError(f"Failed to generate translation: {e}")
        
        response['translation'] = generated_text
        return response



# usage 
if __name__ == "__main__":
    import os 
    from dotenv import load_dotenv
    load_dotenv()
    # Example usage
    service = TranslationService(
        llm_type='google',
        llm_api_key = os.getenv('GOOGLE_API_KEY')
    ) 
    # A german paragraph to translate
    pdf_text = "Dies ist ein Beispieltext auf Deutsch, der ins Englische übersetzt werden soll."
    ins = "You are a document content translator."
    model_name = "gemini-2.0-flash"
    prompt_inputs = {
        "source_language": "german",
        "target_language": "english",
        "pdf_text": pdf_text
    }
    response = service.translate(
        llm_model_name=model_name,
        instructions=ins,
        prompt_inputs=prompt_inputs
    )
    print(response)

