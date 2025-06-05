import os
from typing import Optional, Dict, Any
import yaml

# NOTE: Ensure that the 'prompt_config.yaml' file exists in the same directory as this script or provide the correct path.
# -- Always use this config file to fetch prompts.

class PromptService:
    def __init__(self, config_path: str = "prompt_config.yaml") -> None:
        try:
            # Dir path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, config_path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(f"Error: {e}")
            self.config = {}

    def list_prompts(self) -> list:
        """List all available prompts from the configuration."""
        return self.config.get('prompt_list', {})

    def get_prompt_inputs(self, file_path: str) -> list:
        """Get the list of inputs required for a specific prompt file."""
        return self.config.get('prompt_inputs').get(file_path, [])

    def fetch(self, prompt_category: str, prompt_worker: str, prompt_type: str, inputs: dict) -> str:
        """
        Fetch a specific prompt based on category, worker, and type.

        # NOTE: To use this function appropriately, take reference from the prompt_config.yaml file to understand the structure of prompts.
        # The prompt_category, prompt_worker, and prompt_type should match the keys in the configuration file.
        """
        if not self.config:
            raise ValueError("Configuration not loaded. Please check the config file path.")
        
        # Fetch the prompt file based on the provided category, worker, and type
        try:
            prompt_file = self.config['prompts'][prompt_category][prompt_worker][prompt_type]
        except Exception as e:
            raise ValueError(f"Error accessing prompt configuration: {e}. Available Configurations: {self.config.get('prompts', {})}")
        
        # Build the prompt with the provided inputs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(script_dir, prompt_file)
        try:
            with open(prompt_file_path, 'r') as file:
                prompt_template = file.read()
        except Exception as e:
            raise IOError(f"Error reading the prompt file: {e}. File path: {prompt_file}")
        
        # Validate inputs 
        required_inputs = self.get_prompt_inputs(prompt_file)
        for input_name in required_inputs:
            if input_name not in inputs:
                raise ValueError(
                    f"Missing required input: {input_name}, for prompt file: {prompt_file}. "
                    f"\nRequired inputs are: {required_inputs}"
                )

        # Replace placeholders in the prompt template with actual inputs
        prompt = prompt_template.format(**inputs)
        return prompt




# Example usage:
if __name__ == "__main__":
    prompt_service = PromptService()
    
    # List available prompts
    print("Available Prompts:", prompt_service.list_prompts())
    # Fetch a specific prompt
    try:
        prompt = prompt_service.fetch(
            prompt_category='summarization',
            prompt_worker='default',
            prompt_type='summary',
            inputs={'pdf_text': 'Sample PDF text for summarization.'}
        )
        print("Fetched Prompt:", prompt)
    except Exception as e:
        print(f"Error fetching prompt: {e}")

        
        
        

        
