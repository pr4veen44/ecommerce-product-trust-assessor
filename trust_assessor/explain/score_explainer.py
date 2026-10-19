"""Plain-language explanation of how a Trust Score was calculated (exact decomposition, not an approximation)."""

from __future__ import annotations

from typing import Any

from trust_assessor.scoring.config import COMPONENT_DESCRIPTIONS


def decomposition(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows describing each component's contribution, from a stored assessment dict."""
    rows = []
    for c in assessment["components"]:
        rows.append({
            "key": c["key"], "label": c["label"], "available": c["available"],
            "score": c["score"], "configured_weight": c["weight"], "effective_weight": c["effective_weight"],
            "contribution": c["contribution"], "evidence": c["evidence"] or c.get("missing_reason") or "",
            "description": COMPONENT_DESCRIPTIONS.get(c["key"], ""),
        })
    return rows


def explain_text(assessment: dict[str, Any]) -> list[str]:
    """Step-by-step narrative that reproduces the final number."""
    lines: list[str] = []
    comps = assessment["components"]
    avail = [c for c in comps if c["available"] and c["weight"] > 0]
    lines.append(
        f"1. Each of the {len(comps)} components is normalised to a 0-100 score from the product's data."
    )
    cov = assessment["evidence_coverage"]
    missing = [c["label"] for c in comps if not c["available"] and c["weight"] > 0]
    if missing:
        lines.append(f"2. {len(missing)} component(s) lack data ({', '.join(missing)}). They are excluded and the remaining "
                     f"weights are re-normalised to sum to 100%. Evidence coverage is {cov:.0%}.")
    else:
        lines.append(f"2. All components have data, so evidence coverage is {cov:.0%} and the configured weights apply unchanged.")
    if assessment.get("trust_score") is None:
        lines.append("3. Evidence was too thin to produce a score (see warnings).")
        return lines
    terms = " + ".join(f"{c['effective_weight']:.1%} × {c['score']:.0f}" for c in avail)
    raw = assessment.get("raw_score")
    lines.append(f"3. Weighted average = {terms} = {raw if raw is not None else assessment['trust_score']:.1f}.")
    if assessment.get("adjustments"):
        lines.append("4. " + " ".join(assessment["adjustments"]))
        lines.append(f"5. Final Trust Score = {assessment['trust_score']:.1f}.")
    else:
        lines.append(f"4. No caps or adjustments applied. Final Trust Score = {assessment['trust_score']:.1f}.")
    return lines
