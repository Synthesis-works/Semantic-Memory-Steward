import os
import json
from pydantic import ValidationError
from .models import SemanticAnalysisResult
from strands import Agent


def translate_strands_event(trace, document, **kwargs):
    """Forward a real Strands SDK event into the execution trace.

    Only tool invocations with an identifiable name are recorded (as
    "Tool invoked: <name>"); model output, reasoning text, and raw
    payloads are deliberately ignored. Never raises: tracing must not
    break inference. Active only when a Strands-backed provider executes.
    """
    if trace is None:
        return
    try:
        name = None
        tool_use = kwargs.get("current_tool_use")
        if isinstance(tool_use, dict):
            name = tool_use.get("name")
        if not name:
            nested = kwargs.get("event", {})
            if isinstance(nested, dict):
                tool_use = (nested.get("contentBlockStart", {})
                            .get("start", {}).get("toolUse"))
                if isinstance(tool_use, dict):
                    name = tool_use.get("name")
        if isinstance(name, str) and name:
            trace.record("AI_ANALYSIS", f"Tool invoked: {name}",
                         status="RUNNING", document=document,
                         event_type="tool_use", metadata={"tool": name})
    except Exception:
        pass

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

    def analyze_file(self, file_key: str, content: str, metadata: dict = None, trace=None) -> SemanticAnalysisResult:
        """
        Analyze a file's content and metadata to produce a structured governance result.

        Args:
            file_key: The identifier for the file (e.g. S3 key)
            content: The text content of the file
            metadata: Optional dictionary of file metadata (e.g. size, created_at)
            trace: Optional TraceRecorder. On Strands-backed providers, real
                SDK tool-use events are forwarded into it. The manual REST
                fallback emits nothing here (the pipeline owns stage events).

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

        provider = os.getenv("SMS_LLM_PROVIDER", "bedrock").lower()

        if provider == "bedrock":
            # Normal Strands/Bedrock execution
            original_handler = self.agent.callback_handler
            if trace is not None:
                def _trace_callback(**kwargs):
                    translate_strands_event(trace, file_key, **kwargs)
                self.agent.callback_handler = _trace_callback
            try:
                result = self.agent.structured_output(SemanticAnalysisResult, prompt)
                result.key = file_key  # Ensure key matches
                return result
            except Exception as e:
                raise ValueError(f"Bedrock/Strands inference failed: {e}")
            finally:
                self.agent.callback_handler = original_handler
        elif provider == "sagemaker":
            # Native Strands/SageMaker execution
            from strands.models.sagemaker import SageMakerAIModel
            endpoint_name = os.getenv("SMS_SAGEMAKER_ENDPOINT")
            if not endpoint_name:
                raise ValueError("SMS_SAGEMAKER_ENDPOINT environment variable must be set for sagemaker provider.")
            
            sagemaker_model = SageMakerAIModel(
                endpoint_config={'endpoint_name': endpoint_name, 'region_name': 'us-east-1'},
                payload_config={'max_tokens': 2048, 'temperature': 0.1}
            )
            # Reconfigure the agent to use this specific model for this request
            self.agent.model = sagemaker_model
            original_handler = self.agent.callback_handler
            if trace is not None:
                def _trace_callback(**kwargs):
                    translate_strands_event(trace, file_key, **kwargs)
                self.agent.callback_handler = _trace_callback
            try:
                result = self.agent.structured_output(SemanticAnalysisResult, prompt)
                result.key = file_key
                return result
            except Exception as e:
                raise ValueError(f"SageMaker/Strands inference failed: {e}")
            finally:
                self.agent.callback_handler = original_handler
                
        # AgentCore: run the Strands classification step inside an existing
        # AgentCore Harness (invoked from SMS; the harness is never created
        # here). Same product contract as the Strands/Bedrock path.
        if provider == "agentcore":
            from .agentcore import AgentCoreError, analyze_file_with_harness
            try:
                return analyze_file_with_harness(
                    file_key,
                    content,
                    metadata,
                    system_prompt=self.SYSTEM_PROMPT,
                )
            except AgentCoreError as e:
                raise ValueError(f"AgentCore/Strands inference failed: {e}")

        # Fallback manual parsing for other external test providers
        if provider == "gemini":
            response_text = self._call_gemini(prompt)
        elif provider == "groq":
            response_text = self._call_groq(prompt)
        elif provider == "mistral":
            response_text = self._call_mistral(prompt)
        elif provider == "nvidia":
            response_text = self._call_nvidia(prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Parse the JSON response into our Pydantic model for non-Strands providers
        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]

            data = json.loads(clean_text.strip())

            # The model requires 'retain', 'archive', 'review', or 'delete'.
            # Ensure case-insensitivity from the LLM.
            if "recommended_action" in data and isinstance(data["recommended_action"], str):
                data["recommended_action"] = data["recommended_action"].lower()

            # Ensure the key matches the requested file
            data["key"] = file_key

            return SemanticAnalysisResult(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse agent response into SemanticAnalysisResult: {e}\nResponse: {response_text}")

    def _call_groq(self, prompt: str) -> str:
        """Temporary external LLM adapter using Groq."""
        import urllib.request
        import urllib.error

        api_key = os.getenv("SMS_EXTERNAL_API_KEY")
        if not api_key:
            raise ValueError("SMS_EXTERNAL_API_KEY is not set but provider is groq.")

        model = os.getenv("SMS_EXTERNAL_MODEL", "llama3-8b-8192")
        url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                try:
                    return result["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise ValueError(f"Unexpected response format from Groq: {result}")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            raise RuntimeError(f"External API error: {e.code} - {err_msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error contacting external API: {e.reason}")

    def _call_nvidia(self, prompt: str) -> str:
        """Temporary external LLM adapter using NVIDIA."""
        import urllib.request
        import urllib.error

        api_key = os.getenv("SMS_EXTERNAL_API_KEY")
        if not api_key:
            raise ValueError("SMS_EXTERNAL_API_KEY is not set but provider is nvidia.")

        model = os.getenv("SMS_EXTERNAL_MODEL", "meta/llama-3.1-70b-instruct")
        url = "https://integrate.api.nvidia.com/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                try:
                    return result["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise ValueError(f"Unexpected response format from NVIDIA: {result}")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            raise RuntimeError(f"External API error: {e.code} - {err_msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error contacting external API: {e.reason}")

    def _call_mistral(self, prompt: str) -> str:
        """Temporary external LLM adapter using Mistral."""
        import urllib.request
        import urllib.error

        api_key = os.getenv("SMS_EXTERNAL_API_KEY")
        if not api_key:
            raise ValueError("SMS_EXTERNAL_API_KEY is not set but provider is mistral.")

        model = os.getenv("SMS_EXTERNAL_MODEL", "mistral-small-latest")
        url = "https://api.mistral.ai/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                try:
                    return result["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise ValueError(f"Unexpected response format from Mistral: {result}")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            raise RuntimeError(f"External API error: {e.code} - {err_msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error contacting external API: {e.reason}")

    def _call_gemini(self, prompt: str) -> str:
        """Temporary external LLM adapter using Gemini."""
        import urllib.request
        import urllib.error

        api_key = os.getenv("SMS_EXTERNAL_API_KEY")
        if not api_key:
            raise ValueError("SMS_EXTERNAL_API_KEY is not set but provider is gemini.")

        model = os.getenv("SMS_EXTERNAL_MODEL", "gemini-3.6-flash")
        # API keys travel as ?key= regardless of prefix. (AI Studio now
        # issues AQ.*-format keys; verified live that ?key= is the correct
        # transport and Bearer is rejected for them.)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}

        # We can extract the schema from SemanticAnalysisResult
        # However, Gemini's responseSchema is OpenAPI 3.0, which is very similar to JSON Schema
        # We'll just define the expected schema manually for strictness and compatibility
        schema = {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "category": {"type": "string"},
                "sensitivity": {"type": "string", "enum": ["public", "internal", "confidential", "restricted"]},
                "importance_score": {"type": "number"},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
                "recommended_action": {"type": "string", "enum": ["retain", "archive", "review", "delete"]}
            },
            "required": ["category", "sensitivity", "importance_score", "confidence", "reasoning", "recommended_action"]
        }

        payload = {
            "systemInstruction": {
                "parts": [{"text": self.SYSTEM_PROMPT}]
            },
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    raise ValueError(f"Unexpected response format from Gemini: {result}")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            raise RuntimeError(f"External API error: {e.code} - {err_msg}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error contacting external API: {e.reason}")
