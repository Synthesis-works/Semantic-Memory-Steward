"""
Execution-event trace for the SMS dashboard.

An ExecutionEvent records that something REALLY happened during a pipeline
run or an action execution: which stage, what was observed, and whether it
succeeded, failed, or is waiting on a human. The recorder never invents
events — it only stores what the pipeline/action code explicitly records.

Deliberately absent: model chain-of-thought, raw AWS payloads, credentials,
stack traces. Details are concise, user-safe decision explanations derived
from real outputs.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Stage = Literal[
    "DISCOVERY",
    "CONTENT",
    "AI_ANALYSIS",
    "ENRICHMENT",
    "IMPORTANCE",
    "POLICY",
    "MEMORY",
    "ACTION",
    "HUMAN_APPROVAL",
]

Status = Literal["RUNNING", "SUCCESS", "WAITING", "FAILED"]


class ExecutionEvent(BaseModel):
    """One recorded, real execution event."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    stage: Stage
    event_type: str = "update"
    title: str
    detail: str = ""
    status: Status = "RUNNING"
    document: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceRecorder:
    """Ordered sink for execution events. Starts empty, stays truthful."""

    def __init__(self) -> None:
        self.events: List[ExecutionEvent] = []

    def record(
        self,
        stage: Stage,
        title: str,
        detail: str = "",
        status: Status = "RUNNING",
        document: Optional[str] = None,
        event_type: str = "update",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            stage=stage,
            event_type=event_type,
            title=title,
            detail=detail,
            status=status,
            document=document,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def succeed(
        self,
        stage: Stage,
        title: str,
        detail: str = "",
        document: Optional[str] = None,
        event_type: str = "completed",
    ) -> ExecutionEvent:
        return self.record(stage, title, detail, status="SUCCESS",
                           document=document, event_type=event_type)

    def fail(
        self,
        stage: Stage,
        title: str,
        detail: str = "",
        document: Optional[str] = None,
        event_type: str = "failed",
    ) -> ExecutionEvent:
        return self.record(stage, title, detail, status="FAILED",
                           document=document, event_type=event_type)

    def wait(
        self,
        stage: Stage,
        title: str,
        detail: str = "",
        document: Optional[str] = None,
        event_type: str = "waiting",
    ) -> ExecutionEvent:
        return self.record(stage, title, detail, status="WAITING",
                           document=document, event_type=event_type)

    def clear(self) -> None:
        self.events = []

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [e.model_dump(mode="json") for e in self.events]

    @classmethod
    def from_dicts(cls, data: List[Dict[str, Any]]) -> "TraceRecorder":
        rec = cls()
        rec.events = [ExecutionEvent(**item) for item in (data or [])]
        return rec


_MARKERS = {
    "RUNNING": "◉",
    "SUCCESS": "✓",
    "WAITING": "⏸",
    "FAILED": "✗",
}


def render_trace_lines(events: List[ExecutionEvent]) -> List[str]:
    """Render events as plain status lines. No animation, no invention."""
    lines = []
    for event in events:
        marker = _MARKERS.get(event.status, "•")
        line = f"{marker} {event.title}"
        if event.detail:
            line += f"\n  {event.detail}"
        lines.append(line)
    return lines
