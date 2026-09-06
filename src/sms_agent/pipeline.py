import os
from typing import Dict, Any

from .s3_inventory import S3InventoryCollector
from .s3_content import S3ContentReader
from .agent import SMSAgent
from .importance import ImportanceScorer
from .relationships import RelationshipAnalyzer
from .policy import PolicyEngine
from .models import PolicyDecision, ActionRequest
from .actions import ActionEngine

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
        self.relationship_analyzer = RelationshipAnalyzer()
        self.policy = PolicyEngine()
        self.action_engine = ActionEngine(s3_client=s3_client)

    def process_object(self, file_key: str, skip_inference: bool = True, human_approved: bool = False, execute_action: bool = True) -> Dict[str, Any]:
        """
        Processes a single S3 object through the pipeline.

        Args:
            file_key: S3 object key
            skip_inference: If True, halts before invoking the Bedrock model to avoid errors.
            human_approved: Explicit human authorization for mutating actions.
            execute_action: If True, passes the request to the ActionEngine. If False, returns the request without executing.
        """
        # 1. Observe metadata
        # Fetch inventory to find target and candidate files
        items = self.inventory.collect()

        # Find exact match
        metadata = next((m for m in items if m.key == file_key), None)
        if not metadata:
            raise ValueError(f"Object metadata not found in bucket {self.bucket_name} for key: {file_key}")

        # 2. Retrieve content
        content = self.reader.get_text(metadata)

        # 3. Analyze relationships
        candidates = [item for item in items if item.key != file_key]
        relationships = self.relationship_analyzer.analyze(metadata, content.content if content else None, candidates)

        # 4. Construct input payload for the Strands Agent
        metadata_dict = metadata.model_dump(mode="json")
        relationships_list = [r.model_dump(mode="json") for r in relationships]

        if skip_inference:
            # We can still calculate partial importance based solely on metadata
            partial_importance = self.scorer.score(metadata, None, relationships)

            return {
                "status": "READY_FOR_INFERENCE",
                "bucket": self.bucket_name,
                "key": file_key,
                "metadata": metadata_dict,
                "relationships": relationships_list,
                "content_preview": content.content[:100] + "..." if len(content.content) > 100 else content.content,
                "content_size": content.size_bytes,
                "target_model": self.agent.model_id,
                "partial_importance_score": partial_importance.model_dump(mode="json"),
                "note": "Stopped before Bedrock invocation as requested."
            }

        # 4. Invoke Semantic Analysis
        analysis = self.agent.analyze_file(
            file_key=file_key,
            content=content.content,
            metadata=metadata_dict
        )

        # 6. Calculate Deterministic Importance Score
        importance = self.scorer.score(metadata, analysis, relationships)

        # 7. Make Policy Decision
        decision = self.policy.evaluate(analysis, metadata)

        # 8. Request Action
        requested_action = decision.action
        if requested_action == "TRASH":
            requested_action = "QUARANTINE" # Map domain TRASH to action QUARANTINE

        action_req = ActionRequest(
            s3_uri=metadata.s3_uri or metadata.key,
            bucket=self.bucket_name,
            key=file_key,
            requested_action=requested_action,
            reason=" | ".join(decision.reasons),
            risk=decision.risk,
            human_approved=human_approved
        )

        result_dict = {
            "status": "ANALYZED",
            "bucket": self.bucket_name,
            "key": file_key,
            "metadata": metadata_dict,
            "relationships": relationships_list,
            "analysis": analysis.model_dump(mode="json"),
            "importance": importance.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "action_request": action_req.model_dump(mode="json")
        }

        if execute_action:
            action_result = self.action_engine.execute(action_req)
            result_dict["action_result"] = action_result.model_dump(mode="json")
            result_dict["status"] = "COMPLETED"

        return result_dict
