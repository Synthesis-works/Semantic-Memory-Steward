"""
Small, honest, optional economic-gate for SMS recommendation.

WHY THIS IS SMALL AND HONEST (enforced by tests/test_economics.py):
- SMS asks a second question after the safety/policy decision: "is it even
  worth spending compute to chase this storage?" This module answers that
  question as a *deterministic estimate*, never as a bill.
- No live AWS billing/cost APIs are ever called. All inputs are passed it
  explicitly (bytes, assumed per-file processing cost) and the only price
  constant is the canonical S3 Standard figure already used by
  sms_agent.impact (0.023 USD/GB-month) — no invented numbers.
- The status is one of: WORTHWHILE, NOT_WORTHWHILE, UNKNOWN. Deterministic,
  hermetically testable.
- It can NEVER authorize an action, never override the policy engine, never
  bypass human approval, and never unlock a DELETE. It is purely an
  informational gate layered ON TOP of the existing safety/approval path.

The module makes NO network calls in its decision path (urllib/boto3 are
never touched when computing an assessment); only a caller that explicitly
requests provider inference does so, and the hermetic suite never does.
"""
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

from .impact import S3_STANDARD_USD_PER_GB_MONTH

#: Canonical illustrative SMS per-file processing cost assumed when the
#: caller does not supply one. Labeled an ASSUMPTION (deliberately small;
#: a single Nova Micro round-trip). Overridable by env SMS_ECONOMIC_PROCESSING_USD_PER_FILE.
DEFAULT_PROCESSING_USD_PER_FILE = 0.0005

#: If the estimated saving from a decision is smaller than this, SMS says
#: "not worth compute". Stop-gap, so a sub-penny saving never triggers
#: resource spend.  (Illustrative, configurable via SMS_ECONOMIC_MIN_USD.)
DEFAULT_MIN_SAVING_FOR_WORTHWHILE_USD = 0.01


class EconomicAssessment(BaseModel):
    """Deterministic, labeled-estimate economic gate for a recommendation.

    Every field except the assessment status is either passed in from real
    inputs or derived deterministically from them. No live billing/savings
    numbers are fetched. status/estimated_* are always labeled estimates.
    """
    status: Literal["WORTHWHILE", "NOT_WORTHWHILE", "UNKNOWN"]
    estimated_benefit_usd: float = Field(ge=0.0, description="Labeled ESTIMATE.")
    estimated_processing_cost_usd: float = Field(
        ge=0.0, description="Assumption-based ESTIMATE.")
    estimated_net_benefit_usd: float = Field(
        description="benefit - cost. Labeled ESTIMATE.")
    reasoning: str = Field(
        description="Concise, honest, user-facing explanation.")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicit labels so nobody mistakes this for the AWS bill.")


def compute_economic_assessment(
    potential_saving_bytes: Optional[int],
    processing_cost_usd: Optional[float] = None,
    min_saving_for_worthwhile_usd: Optional[float] = None,
    s3_standard_usd_per_gb_month: float = S3_STANDARD_USD_PER_GB_MONTH,
) -> EconomicAssessment:
    """Determine whether chasing a storage action is economically worthwhile.

    Pure + deterministic: given the same bytes and costs it always returns
    the same assessment. Makes no AWS callsgrave; prices are estimates.

    Args:
        potential_saving_bytes: bytes SMS could stop storing in the managed
            path (only genuine ARCHIVE policy candidates count as savings —
            REVIEW is never counted). None/missing → UNKNOWN.
        processing_cost_usd: assumed SMS cost to process the file. If None,
            DEFAULT_PROCESSING_USD_PER_FILE + env SMS_ECONOMIC_PROCESSING_USD_PER_FILE.
        min_saving_for_worthwhile_usd: floor below which a positive saving
            is still considered NOT worth the compute. Defaults to
            DEFAULT_MIN_SAVING_FOR_WORTHWHILE_USD.
        s3_standard_usd_per_gb_month: canonical pricing figure; defaults to
            the repo's single canonical S3 Standard value.
    """
    import os
    gb = 1024 ** 3
    pct_floor = (min_saving_for_worthwhile_usd
                 if min_saving_for_worthwhile_usd is not None
                 else float(os.getenv("SMS_ECONOMIC_MIN_USD",
                                      str(DEFAULT_MIN_SAVING_FOR_WORTHWHILE_USD))))
    proc = (processing_cost_usd
            if processing_cost_usd is not None
            else float(os.getenv("SMS_ECONOMIC_PROCESSING_USD_PER_FILE",
                                 str(DEFAULT_PROCESSING_USD_PER_FILE))))

    if potential_saving_bytes is None or potential_saving_bytes < 0:
        return EconomicAssessment(
            status="UNKNOWN",
            estimated_benefit_usd=0.0,
            estimated_processing_cost_usd=max(proc, 0.0),
            estimated_net_benefit_usd=0.0,
            reasoning=("SMS cannot estimate the potential saving for this "
                       "document (no reliable size/savings input). No "
                       "economic claim is made."),
            assumptions=_assumptions(weight_bytes=None),
        )

    benefit = (potential_saving_bytes / gb) * s3_standard_usd_per_gb_month
    cost = max(proc, 0.0)
    net = benefit - cost

    if benefit <= 0:
        return EconomicAssessment(
            status="NOT_WORTHWHILE",
            estimated_benefit_usd=benefit,
            estimated_processing_cost_usd=cost,
            estimated_net_benefit_usd=net,
            reasoning=("No genuine storage benefit was identified for this "
                       "candidate (zero or negative potential saving). SMS "
                       "recommends not spending compute on it."),
            assumptions=_assumptions(weight_bytes=potential_saving_bytes),
        )
    if net < pct_floor:
        return EconomicAssessment(
            status="NOT_WORTHWHILE",
            estimated_benefit_usd=benefit,
            estimated_processing_cost_usd=cost,
            estimated_net_benefit_usd=net,
            reasoning=(
                f"Estimated saving ~${benefit:.4f}/mo is below the "
                f"~${pct_floor:.4f}/mo cost floor for spending compute "
                f"(illustrative). SMS: NOT worth it to chase."),
            assumptions=_assumptions(weight_bytes=potential_saving_bytes),
        )
    if net == 0 or abs(net) < 1e-12:
        return EconomicAssessment(
            status="NOT_WORTHWHILE",
            estimated_benefit_usd=benefit,
            estimated_processing_cost_usd=cost,
            estimated_net_benefit_usd=net,
            reasoning=("Estimated saving equals estimated processing cost; "
                       "no net benefit. SMS: NOT worth it."),
            assumptions=_assumptions(weight_bytes=potential_saving_bytes),
        )
    return EconomicAssessment(
        status="WORTHWHILE",
        estimated_benefit_usd=benefit,
        estimated_processing_cost_usd=cost,
        estimated_net_benefit_usd=net,
        reasoning=(
            f"Estimated storage saving ~${benefit:.4f}/mo exceeds the "
            f"~${cost:.4f}/mo estimated processing cost (net ~${net:.4f}/mo). "
            f"SMS: economical to pursue (still governed by policy/safety)."),
        assumptions=_assumptions(weight_bytes=potential_saving_bytes),
    )


def _assumptions(weight_bytes: Optional[int]) -> List[str]:
    parts = [
        "ESTIMATE: deterministic model, not the AWS bill.",
        "S3 Standard pricing is the single canonical S3 Standard rate "
        f"(${S3_STANDARD_USD_PER_GB_MONTH}/GB-month), reused from impact.py.",
        "Processing cost is an explicit assumption, not measured.",
    ]
    if weight_bytes is None:
        parts.append("Inputs were insufficient; no claim is made.")
    return parts
