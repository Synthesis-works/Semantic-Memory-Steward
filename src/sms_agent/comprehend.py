import boto3
import logging
from typing import Dict, Any, List

log = logging.getLogger(__name__)

class ComprehendAnalyzer:
    """Enrichment layer using Amazon Comprehend for entity and PII detection.
    
    This abstracts away the AWS SDK calls and ensures raw sensitive text
    is not returned or persisted, only metadata (type and confidence).
    """
    
    def __init__(self, client=None):
        """Initialize with an optional boto3 client (for testing/mocking)."""
        self.client = client or boto3.client('comprehend', region_name='us-east-1')
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and return structured metadata without raw PII.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Dict containing lists of entities and PII metadata.
        """
        if not text or not text.strip():
            return {"entities": [], "pii_entities": [], "errors": []}
            
        # Comprehend has a 5000 byte limit for english text analysis.
        # For our MVP enrichment, we'll analyze the first 4900 bytes to be safe.
        encoded = text.encode('utf-8')
        if len(encoded) > 4900:
            text = encoded[:4900].decode('utf-8', errors='ignore')
            
        result = {
            "entities": [],
            "pii_entities": [],
            "errors": []
        }
        
        try:
            # 1. Detect standard entities (Organizations, Locations, etc.)
            entities_resp = self.client.detect_entities(Text=text, LanguageCode='en')
            for ent in entities_resp.get('Entities', []):
                # Only keep type and confidence, omit 'Text'
                result["entities"].append({
                    "type": ent.get('Type'),
                    "score": ent.get('Score', 0.0)
                })
        except Exception as e:
            log.warning(f"Comprehend DetectEntities failed: {e}")
            result["errors"].append(f"DetectEntities failed: {str(e)}")
            
        try:
            # 2. Detect PII entities (Email, Phone, SSN, etc.)
            pii_resp = self.client.detect_pii_entities(Text=text, LanguageCode='en')
            for pii in pii_resp.get('PiiEntities', []):
                # Omit actual PII text/offsets
                result["pii_entities"].append({
                    "type": pii.get('Type'),
                    "score": pii.get('Score', 0.0)
                })
        except Exception as e:
            log.warning(f"Comprehend DetectPiiEntities failed: {e}")
            result["errors"].append(f"DetectPiiEntities failed: {str(e)}")
            
        return result
