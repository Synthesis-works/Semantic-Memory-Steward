import os
from typing import Dict, Any

from .s3_inventory import S3InventoryCollector
from .s3_content import S3ContentReader
from .agent import SMSAgent
from .importance import ImportanceScorer
from .policy import PolicyEngine
from .models import PolicyDecision

class SMSPipeline:
    """End-to-end processing pipeline for Semantic Memory Steward."""

    def __init__(self, bucket_name: str, s3_client=None):
        self.bucket_name = bucket_name
        self.s3_client = s3_client
        
        # Initialize the domain layers
        self.inventory = S3InventoryCollector(bucket_name, s3_client=s3_client)
        self.reader = S3ContentReader(s3_client=s3_client)
        self.agent = SMSAgent()
        self.scorer = ImportanceScorer()
        self.policy = PolicyEngine()

    def process_object(self, file_key: str, skip_inference: bool = True) -> Dict[str, Any]:
        """
        Processes a single S3 object through the pipeline.
        
        Args:
            file_key: S3 object key
            skip_inference: If True, halts before invoking the Bedrock model to avoid errors.
        """
        # 1. Observe metadata
        # We temporarily set the prefix to narrow the inventory search
        original_prefix = self.inventory.prefix
        self.inventory.prefix = file_key
        try:
            items = self.inventory.collect()
        finally:
            self.inventory.prefix = original_prefix
            
        # Find exact match
        metadata = next((m for m in items if m.key == file_key), None)
        if not metadata:
            raise ValueError(f"Object metadata not found in bucket {self.bucket_name} for key: {file_key}")

        # 2. Retrieve content
        content = self.reader.get_text(metadata)

        # 3. Construct input payload for the Strands Agent
        metadata_dict = metadata.model_dump(mode="json")
        
        if skip_inference:
            # We can still calculate partial importance based solely on metadata
            partial_importance = self.scorer.score(metadata, None)
            
            return {
                "status": "READY_FOR_INFERENCE",
                "bucket": self.bucket_name,
                "key": file_key,
                "metadata": metadata_dict,
                "content_preview": content.content[:100] + "..." if len(content.content) > 100 else content.content,
                "content_size": content.size_bytes,
                "target_model": self.agent.model_id,
                "partial_importance_score": partial_importance.model_dump(mode="json"),
                "note": "Stopped before Bedrock invocation as requested."
            }
            
        # 4. Invoke Semantic Analysis (CURRENTLY BLOCKED BY AWS)
        analysis = self.agent.analyze_file(
            file_key=file_key,
            content=content.content,
            metadata=metadata_dict
        )
        
        # 5. Calculate Deterministic Importance Score
        importance = self.scorer.score(metadata, analysis)
        
        # 6. Make Policy Decision
        decision = self.policy.evaluate(analysis, metadata)
        
        return {
            "status": "COMPLETED",
            "bucket": self.bucket_name,
            "key": file_key,
            "analysis": analysis.model_dump(mode="json"),
            "importance": importance.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json")
        }
