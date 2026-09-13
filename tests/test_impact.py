"""Impact model tests: every number must derive from real inputs.

No historical data may be fabricated, REVIEW documents are never
counted as savings, cost is always an explicitly-assumed estimate,
and projections are flat, labeled, and assumption-bound.
"""
from sms_agent import impact
from sms_agent.impact import (
    S3_STANDARD_USD_PER_GB_MONTH,
    compute_impact,
    format_bytes,
    projection_points,
    scale_scenario,
)


def _row(filename, size, policy):
    return {"Filename": filename, "SizeBytes": size, "Policy": policy}


def test_total_bytes_from_real_sizes():
    model = compute_impact(
        [_row("demo/a.txt", 1000, "retain"),
         _row("demo/b.txt", 2000, "review")],
        managed_sizes=[])
    assert model["doc_count"] == 2
    assert model["total_bytes"] == 3000


def test_potential_only_counts_archive_policy():
    model = compute_impact(
        [_row("demo/keep.txt", 1000, "retain"),
         _row("demo/review.txt", 2000, "review"),
         _row("demo/old.txt", 4000, "archive")],
        managed_sizes=[])
    assert model["potential_bytes"] == 4000
    assert model["potential_docs"] == ["demo/old.txt"]
    assert model["review_docs"] == 1
    assert model["potential_pct"] == 4000 / 7000 * 100


def test_managed_bytes_are_actual_relocated_storage():
    model = compute_impact([_row("demo/a.txt", 1000, "retain")],
                           managed_sizes=[500, 1500])
    assert model["managed_bytes"] == 2000


def test_cost_uses_explicit_pricing_assumption():
    model = compute_impact([_row("demo/a.txt", 1024 ** 3, "retain")],
                           managed_sizes=[])
    assert model["price_usd_per_gb_month"] == S3_STANDARD_USD_PER_GB_MONTH
    assert model["monthly_cost_usd"] == S3_STANDARD_USD_PER_GB_MONTH
    assert "estimate" in " ".join(model["assumptions"]).lower()
    assert "not" in " ".join(model["assumptions"]).lower()  # not your bill


def test_monthly_savings_derive_from_potential():
    model = compute_impact([_row("demo/old.txt", 1024 ** 3, "archive")],
                           managed_sizes=[])
    assert model["monthly_savings_usd"] == S3_STANDARD_USD_PER_GB_MONTH


def test_format_bytes_boundaries():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 ** 2) == "5.0 MB"
    assert "GB" in format_bytes(3 * 1024 ** 3)


def test_projection_is_flat_labeled_and_bounded():
    points = projection_points(current_bytes=1000, potential_bytes=250,
                               months=12)
    assert len(points) == 13  # month 0..12
    months = [p["month"] for p in points]
    assert months == sorted(months)
    assert all(p["baseline_bytes"] == 1000 for p in points)
    assert all(p["managed_bytes"] == 750 for p in points)
    assert all(p["kind"] == "projected" for p in points)


def test_projection_takes_no_history():
    import inspect
    params = list(inspect.signature(projection_points).parameters)
    assert params == ["current_bytes", "potential_bytes", "months"]


def test_empty_workspace_is_zero_not_fabricated():
    model = compute_impact([], managed_sizes=[])
    assert model["total_bytes"] == 0
    assert model["potential_bytes"] == 0
    assert model["potential_pct"] == 0
    assert model["monthly_cost_usd"] == 0
    points = projection_points(0, 0)
    assert all(p["baseline_bytes"] == 0 and p["managed_bytes"] == 0
               for p in points)


def test_scale_scenario_is_labeled_illustrative():
    scenario = scale_scenario(potential_pct=40.0)
    assert scenario["label_kind"] == "illustrative"
    assert scenario["scale_bytes"] == 100 * 1024 ** 3
    assert scenario["saved_bytes"] == 40 * 1024 ** 3


def test_impact_module_has_no_hardcoded_money():
    import pathlib
    text = pathlib.Path("src/sms_agent/impact.py").read_text(encoding="utf-8")
    assert "$50" not in text
    assert "76%" not in text
