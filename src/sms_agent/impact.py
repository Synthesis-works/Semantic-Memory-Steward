"""
Storage & cost impact model for the SMS dashboard.

Honesty contract (enforced by tests/test_impact.py):
- Every byte figure derives from real inputs (S3 object sizes,
  policy/action state). Nothing is invented.
- Only policy ARCHIVE documents count as *potential* reduction.
  REVIEW documents are never counted as savings.
- Managed (trash/archive) bytes are *actual* relocated storage,
  measured from inventory — still stored, out of the workspace.
- Cost is an explicitly-assumed estimate (illustrative S3 Standard
  pricing), never presented as the user's AWS bill.
- The projection is flat, labeled "projected", and takes no history
  because no historical measurements exist.
"""
from typing import Any, Dict, List

#: Illustrative S3 Standard storage pricing. An assumption, not a bill.
S3_STANDARD_USD_PER_GB_MONTH = 0.023

_GB = 1024 ** 3
#: Illustrative scale scenario size (100 GB workspace).
SCALE_SCENARIO_BYTES = 100 * _GB


def format_bytes(num_bytes: int) -> str:
    """Human-readable sizes: B → KB → MB → GB → TB."""
    value = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover - loop always returns


def _gb(num_bytes: int) -> float:
    return (num_bytes or 0) / _GB


def compute_impact(workspace_rows: List[Dict[str, Any]],
                   managed_sizes: List[int]) -> Dict[str, Any]:
    """Build the impact model from current real inputs.

    workspace_rows: dashboard rows carrying SizeBytes + Policy.
    managed_sizes: byte sizes of current trash/archive inventory objects.
    """
    rows = workspace_rows or []
    total = sum(int(row.get("SizeBytes") or 0) for row in rows)
    archived = [row for row in rows
                if (row.get("Policy") or "").lower() == "archive"]
    potential = sum(int(row.get("SizeBytes") or 0) for row in archived)
    review_docs = sum(1 for row in rows
                      if (row.get("Policy") or "").lower() == "review")
    managed = sum(managed_sizes or [])
    monthly_cost = _gb(total) * S3_STANDARD_USD_PER_GB_MONTH
    monthly_savings = _gb(potential) * S3_STANDARD_USD_PER_GB_MONTH
    return {
        "doc_count": len(rows),
        "total_bytes": total,
        "potential_bytes": potential,
        "potential_docs": [row.get("Filename") for row in archived],
        "potential_pct": (potential / total * 100) if total else 0,
        "review_docs": review_docs,
        "managed_bytes": managed,
        "monthly_cost_usd": monthly_cost,
        "monthly_savings_usd": monthly_savings,
        "price_usd_per_gb_month": S3_STANDARD_USD_PER_GB_MONTH,
        "assumptions": [
            "Cost is an estimate using illustrative S3 Standard pricing "
            f"(${S3_STANDARD_USD_PER_GB_MONTH}/GB-month), not your AWS bill.",
            "Potential reduction counts only ARCHIVE-policy documents; "
            "REVIEW documents are never counted as savings.",
            "Moved bytes remain stored until retention removes them.",
        ],
    }


def projection_points(current_bytes: int, potential_bytes: int,
                      months: int = 12) -> List[Dict[str, Any]]:
    """Flat illustrative projection. No history exists, so no trend is
    invented: the baseline holds current storage, the managed line holds
    current minus potential. Every point is labeled projected."""
    managed = max((current_bytes or 0) - (potential_bytes or 0), 0)
    return [{"month": month,
             "baseline_bytes": current_bytes or 0,
             "managed_bytes": managed,
             "kind": "projected"} for month in range(months + 1)]


def scale_scenario(potential_pct: float,
                   scale_bytes: int = SCALE_SCENARIO_BYTES) -> Dict[str, Any]:
    """Apply the measured potential percentage to an illustrative scale.

    Explicitly labeled illustrative: this is not the user's workspace.
    """
    saved = int(scale_bytes * (potential_pct or 0) / 100)
    return {"label_kind": "illustrative",
            "scale_bytes": scale_bytes,
            "saved_bytes": saved,
            "monthly_savings_usd": (_gb(saved)
                                    * S3_STANDARD_USD_PER_GB_MONTH)}
