from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os


# -- Base Class --
class SummarizationServiceBase(ABC):
    """
    Abstract base class for summarization services.
    """
    @staticmethod
    def validate_input(inputs: Any) -> bool:
        """
        Validate the input data for summarization. 
        """
        pass

    @abstractmethod 
    def summarize(self) -> Dict[str, Any]:
        """
        Summarize the provided inputs using the configured LLM and prompt.
        """
        pass 

    @abstractmethod 
    def summary_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the summary.
        """
        pass


# -- Concrete Implementation --
class SummarizationService(SummarizationServiceBase):
    def __init__(self, llm) -> None: 
        self.llm = llm 

    @staticmethod
    def validate_input(inputs: Any) -> bool:
        """
        Validate the input data for summarization.
        """
        assert isinstance(inputs, dict), "Inputs must be a dictionary." 
        assert isinstance(inputs.get('pdf_text'), str), "Input 'pdf_text' must be a string."
        assert 'pdf_text' in inputs, "Input must contain 'pdf_text' key."
        return True
    
    def summarize(self):
        pass

    



