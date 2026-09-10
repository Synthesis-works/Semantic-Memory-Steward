from typing import Literal
from .models import ActionRequest, ExecutionMode

class ActionAuthorizer:
    """Safety boundary that independently validates action requests."""
    
    def validate(self, request: ActionRequest) -> Literal["AUTHORIZED", "PENDING_APPROVAL", "BLOCKED"]:
        """
        Evaluate whether an action is allowed based strictly on policy rules,
        not LLM reasoning.
        """
        # Rule 1: No destructive DELETE action may execute.
        if request.requested_action == "DELETE":
            return "BLOCKED"
            
        # Rule 4: KEEP never mutates S3, always authorized (as a no-op).
        if request.requested_action == "KEEP":
            return "AUTHORIZED"
            
        # Rule 3: REVIEW always requires human review.
        if request.requested_action == "REVIEW":
            return "PENDING_APPROVAL"
            
        # QUARANTINE / ARCHIVE (Mutating actions)
        if request.requested_action in ("QUARANTINE", "ARCHIVE"):
            # Autonomous low-risk cleanup where policy permits it
            if request.requested_action == "ARCHIVE" and request.risk == "LOW":
                return "AUTHORIZED"

            if request.execution_mode == ExecutionMode.SAFE:
                # Rule 2: A high-sensitivity/high-risk action must require human approval in SAFE mode.
                if request.human_approved:
                    return "AUTHORIZED"
                else:
                    return "PENDING_APPROVAL"
            elif request.execution_mode == ExecutionMode.AUTONOMOUS:
                # High risk/sensitivity requires human approval, but in autonomous mode
                # we don't have a human, so we must safely block it.
                return "BLOCKED"
            elif request.execution_mode == ExecutionMode.TURBO:
                return "BLOCKED"
                
        # Rule 6: Invalid/unknown actions must be rejected.
        return "BLOCKED"
