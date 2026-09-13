"""Hermetic tests for the optional SMS economic gate (sms_agent.economics).

These tests NEVER:
- call a live AWS service (no boto3/boto/urllib/bedrock; no SMS_*_API_KEY,
  no SMS_BEDROCK_REGION, no iam/iam-policy/harness/DynamoDB/S3 are touched),
- read external secrets, or make live billing/savings calls,
- claim numbers economics.py explicitly labels as estimates are real bills,
- override/weaken/bypass the policy engine, human approval, or the
  action/safety path (economics is purely an informational gate ON TOP).

Deterministic + pure: same inputs -> same assessment. Hermetic only.
"""
import os
from unittest.mock import patch

import pytest

from sms_agent.economics import (
    compute_economic_assessment,
    EconomicAssessment,
    DEFAULT_PROCESSING_USD_PER_FILE,
    DEFAULT_MIN_SAVING_FOR_WORTHWHILE_USD,
)

GB = 1024 ** 3
S3_STANDARD_USD_PER_GB_MONTH = 0.023  # canonical figure reused from impact.py


# ---------------------------------------------------------------------------
# Deterministic core (no AWS, no env required)
# ---------------------------------------------------------------------------

def test_assessment_when_saving_clearly_exceeds_cost():
    # 100 GB genuinely ARCHIVE-eligible, tiny assumed processing cost.
    r = compute_economic_assessment(
        potential_saving_bytes=100 * GB,
        processing_cost_usd=DEFAULT_PROCESSING_USD_PER_FILE,
        s3_standard_usd_per_gb_month=S3_STANDARD_USD_PER_GB_MONTH,
    )
    assert r.status == "WORTHWHILE"
    assert r.estimated_benefit_usd == pytest.approx(2.3, rel=0.05)
    assert r.estimated_net_benefit_usd > 0
    assert r.assumptions  # labeled ESTIMATE assumptions are always present


def test_assessment_when_benefit_below_cost_is_NOT_WORTHWHILE():
    # Sub-penny-ish saving vs a fixed per-file processing cost.
    r = compute_economic_assessment(
        potential_saving_bytes=1024,  # 1 KB -> ~0.0000235/mo
        processing_cost_usd=DEFAULT_MIN_SAVING_FOR_WORTHWHILE_USD,
    )
    assert r.status == "NOT_WORTHWHILE"
    # The estimate is NOT presented as a bill.
    assert r.reasoning
    assert any("ESTIMATE" in a.upper() for a in r.assumptions)


def test_assessment_equal_benefit_and_cost_is_NOT_WORTHWHILE():
    # Deterministic boundary: benefit == cost => NOT_WORTHWHILE (never a
    # net-positive claim). This is the honest stop-gap that prevents a
    # zero-net decision from spurring resource spend.
    benefit_bytes = 1024 ** 2                       # ~1 MB
    benefit = (benefit_bytes / GB) * S3_STANDARD_USD_PER_GB_MONTH
    r = compute_economic_assessment(
        potential_saving_bytes=benefit_bytes,
        processing_cost_usd=benefit,  # exactly cancels
    )
    assert r.status == "NOT_WORTHWHILE"
    assert r.estimated_net_benefit_usd == pytest.approx(0.0, abs=1e-9)


def test_assessment_unknown_when_input_missing():
    r = compute_economic_assessment(potential_saving_bytes=None)
    assert r.status == "UNKNOWN"
    assert r.estimated_benefit_usd == 0.0


def test_assessment_rejects_negative_bytes_as_unknown_not_savings():
    r = compute_economic_assessment(potential_saving_bytes=-5)
    assert r.status == "UNKNOWN"
    assert r.estimated_benefit_usd == 0.0


# ---------------------------------------------------------------------------
# Safety invariants: economics NEVER changes policy / approval / actions
# ---------------------------------------------------------------------------

def test_economics_is_pure_and_deterministic():
    kw = dict(potential_saving_bytes=50 * GB)
    a = compute_economic_assessment(**kw)
    b = compute_economic_assessment(**kw)
    assert a.model_dump() == b.model_dump()


def test_economics_makes_no_network_calls():
    # The entire decision path is hermetic: no urllib/boto3 invocation can
    # happen inside economics.compute_economic_assessment. We patch the
    # relevant stdlib/boto names so a regression that starts dialing out
    # fails loudly instead of hitting the network.
    with patch("urllib.request.urlopen", autospec=True) as u, \
         patch("boto3.client", autospec=True) as b:
        r = compute_economic_assessment(potential_saving_bytes=50 * GB)
        assert r.status == "WORTHWHILE"
    u.assert_not_called()
    b.assert_not_called()


def test_economics_never_authorizes_or_overrides_policy():
    # The economic gate can say "NOT worth compute", but that must never
    # show up as a policy/authorization change, an approval bypass, or a
    # new DELETE. We assert the economics API itself contains no surface
    # to mutate policy/output actions: the only exported symbols are the
    # pure function and its deterministic model.
    import sms_agent.economics as econ
    exports = {n for n in dir(econ) if not n.startswith("_")}
    assert {"compute_economic_assessment", "EconomicAssessment"} <= exports
    # No act/authorize/approve/delete surface leaks from economics.
    assert not any("delete" in n.lower() or "authorize" in n.lower()
                   or "approve" in n.lower() for n in exports)


def test_economics_honest_labels_in_reasoning_and_assumptions():
    r = compute_economic_assessment(potential_saving_bytes=50 * GB)
    joined = (r.reasoning + " " + " ".join(r.assumptions)).lower()
    # Nobody mistakes an ESTIMATE for the AWS bill.
    assert "estimate" in joined
