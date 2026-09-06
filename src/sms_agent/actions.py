import boto3
from botocore.exceptions import ClientError
from .models import ActionRequest, ActionResult
from .authorization import ActionAuthorizer

class ActionEngine:
    """Executes actions on S3 after enforcing strict authorization rules."""
    
    def __init__(self, s3_client=None):
        # We explicitly use boto3.client directly here to enforce AWS I/O isolation
        self.s3_client = s3_client or boto3.client("s3")
        self.authorizer = ActionAuthorizer()
        
    def execute(self, request: ActionRequest) -> ActionResult:
        """Execute a requested action safely."""
        
        # 1 & 5: Validate and enforce authorization boundary
        auth_status = self.authorizer.validate(request)
        
        if auth_status == "BLOCKED":
            return ActionResult(
                action=request.requested_action,
                key=request.key,
                status="BLOCKED",
                message="Action explicitly blocked by safety layer."
            )
            
        if auth_status == "PENDING_APPROVAL":
            return ActionResult(
                action=request.requested_action,
                key=request.key,
                status="PENDING_APPROVAL",
                message="Action requires explicit human approval."
            )
            
        # Action is AUTHORIZED beyond this point
        
        # 9, 10, 11: Safely handle non-mutating/blocked actions
        if request.requested_action == "KEEP":
            return ActionResult(
                action="KEEP",
                key=request.key,
                status="VERIFIED_NO_ACTION",
                message="No mutation required."
            )
            
        # QUARANTINE execution
        if request.requested_action == "QUARANTINE":
            return self._execute_quarantine(request)
            
        # Any other action (like ARCHIVE) not fully implemented yet
        return ActionResult(
            action=request.requested_action,
            key=request.key,
            status="FAILED",
            message=f"Executor for action {request.requested_action} is not yet implemented."
        )
        
    def _derive_quarantine_destination(self, source_key: str) -> str:
        """Safely derive the quarantine destination key while retaining directory structure."""
        if not source_key or source_key.strip() == "":
            raise ValueError("Source key cannot be empty.")
            
        # Reject obvious traversal or malformed paths
        if ".." in source_key or "//" in source_key or source_key.startswith("/"):
            raise ValueError("Source key contains unsafe or malformed path characters.")
            
        # Do not quarantine if it's already in trash
        if source_key.startswith("trash/"):
            raise ValueError("Source object is already in the quarantine namespace.")
            
        return f"trash/{source_key}"

    def _execute_quarantine(self, request: ActionRequest) -> ActionResult:
        """
        Move a file to a quarantine prefix (trash/) safely.
        Uses a copy-then-verify-then-delete pattern.
        """
        bucket = request.bucket
        source_key = request.key
        
        try:
            dest_key = self._derive_quarantine_destination(source_key)
        except ValueError as e:
            return ActionResult(
                action="QUARANTINE",
                key=source_key,
                status="FAILED",
                message=str(e)
            )
        
        try:
            # Idempotency and collision checks
            dest_exists = self._object_exists(bucket, dest_key)
            source_exists = self._object_exists(bucket, source_key)
            
            if dest_exists and not source_exists:
                return ActionResult(
                    action="QUARANTINE",
                    key=source_key,
                    status="VERIFIED",
                    message=f"Object already quarantined at {dest_key}."
                )
                
            if dest_exists and source_exists:
                return ActionResult(
                    action="QUARANTINE",
                    key=source_key,
                    status="FAILED",
                    message=f"Quarantine destination {dest_key} is already occupied."
                )
                
            if not source_exists and not dest_exists:
                return ActionResult(
                    action="QUARANTINE",
                    key=source_key,
                    status="FAILED",
                    message="Source object does not exist."
                )
                
            # 1. Execute copy
            copy_source = {'Bucket': bucket, 'Key': source_key}
            self.s3_client.copy_object(CopySource=copy_source, Bucket=bucket, Key=dest_key)
            
            # 2. Verify destination
            if not self._object_exists(bucket, dest_key):
                return ActionResult(
                    action="QUARANTINE",
                    key=source_key,
                    status="FAILED",
                    message="Destination verification failed after copy."
                )
                
            # 3. Remove source
            self.s3_client.delete_object(Bucket=bucket, Key=source_key)
            
            # 4. Verify source state
            if self._object_exists(bucket, source_key):
                return ActionResult(
                    action="QUARANTINE",
                    key=source_key,
                    status="FAILED",
                    message="Source object was not removed after copy."
                )
                
            return ActionResult(
                action="QUARANTINE",
                key=source_key,
                status="VERIFIED",
                message=f"Successfully quarantined to {dest_key}."
            )
            
        except Exception as e:
            return ActionResult(
                action="QUARANTINE",
                key=source_key,
                status="FAILED",
                message=f"Quarantine execution failed: {str(e)}"
            )

    def _object_exists(self, bucket: str, key: str) -> bool:
        """Helper to safely verify object existence."""
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
