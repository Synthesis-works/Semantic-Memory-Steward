import os
import json
from pydantic import ValidationError
from .models import SemanticAnalysisResult
from strands import Agent

class SMSAgent:
    """Semantic Memory Steward Agent."""
    
    SYSTEM_PROMPT = (
        "You are Semantic Memory Steward, a professional data-governance agent. "
        "Analyze a file's content and metadata to determine its semantic category, "
        "sensitivity, importance, and a safe recommended action. Be conservative with "
        "sensitive or important information. Never recommend destructive deletion solely "
        "because a file is old or duplicated. Explain the reasoning. "
        "Respond ONLY with a valid JSON object matching the requested schema."
    )
    
    def __init__(self, model_id: str = None):
        """
        Initialize the SMS agent.
        
        Args:
            model_id: The Bedrock model ID. Defaults to env var SMS_BEDROCK_MODEL_ID or a sensible default.
        """
        self.model_id = model_id or os.getenv("SMS_BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
        
        # Initialize the Strands agent
        self.agent = Agent(
            name="SemanticMemorySteward",
            system_prompt=self.SYSTEM_PROMPT,
            model=self.model_id
        )
        
    def analyze_file(self, file_key: str, content: str, metadata: dict = None) -> SemanticAnalysisResult:
        """
        Analyze a file's content and metadata to produce a structured governance result.
        
        Args:
            file_key: The identifier for the file (e.g. S3 key)
            content: The text content of the file
            metadata: Optional dictionary of file metadata (e.g. size, created_at)
            
        Returns:
            SemanticAnalysisResult: Structured analysis result
        """
        metadata_str = json.dumps(metadata) if metadata else "None"
        
        prompt = (
            f"File Key: {file_key}\n"
            f"Metadata: {metadata_str}\n"
            f"Content:\n{content}\n\n"
            "Analyze the above file and return the JSON result."
        )
        
        # In a real implementation, we would pass the prompt to the Strands Agent
        # and expect it to return the JSON structure.
        response = self.agent(prompt)
        # If response is an object with text or content, we would extract it. 
        # Assuming the call returns the text directly or we can get string representation.
        response_text = str(response)
        
        # Parse the JSON response into our Pydantic model
        try:
            # We strip in case the model returns markdown code blocks like ```json ... ```
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
                
            data = json.loads(clean_text.strip())
            
            # Ensure the key matches the requested file
            data["key"] = file_key
            
            return SemanticAnalysisResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse agent response into SemanticAnalysisResult: {e}\nResponse: {response_text}")
