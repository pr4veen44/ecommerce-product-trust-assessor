"""Transparent Trust Score engine.

The score is a *weighted average of normalised (0-100) component scores*. Components whose
input data is missing are excluded and the remaining weights are re-normalised, which is
reported as "evidence coverage". Nothing is imputed: missing data lowers coverage and
confidence rather than being silently replaced by a default value.

THE SCORE IS A HEURISTIC SUMMARY. It has not been validated against ground-truth labels
of product trustworthiness (none exist in the dataset) and should be used for screening
and exploration, not as proof that a product or review is genuine or fake.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


from trust_assessor.scoring.config import COMPONENT_KEYS, COMPONENT_LABELS, ScoringConfig

# Things the dataset cannot support, always disclosed instead of being guessed.
NOT_ASSESSED = [
    ("Return risk", "The dataset contains no return, refund or defect data, so no return-risk estimate is made."),
    ("Seller reliability", "Only a store/brand name is available. There are no seller ratings, fulfilment or dispute data, so seller reliability is not scored."),
    ("Counterfeit / product safety", "No authenticity-of-goods or safety-recall information is available."),
    ("Price fairness", "Only a single listed price (when present) is available, with no price history or comparison."),
]


@dataclass
class ComponentResult:
    key: str
    label: str
    weight: float                 # normalised weight configured for this component (sums to 1 over all)
    score: float | None           # 0-100, None if unavailable
    available: bool
    evidence: str                 # one-line, number-backed explanation
    details: dict[str, Any] = field(default_factory=dict)
    missing_reason: str | None = None
    effective_weight: float = 0.0      # weight after re-normalising over available components
    contribution: float = 0.0          # points contributed to the final score


@dataclass
class TrustAssessment:
    parent_asin: str
    trust_score: float | None
    grade: str
    confidence: str
    evidence_coverage: float
    n_reviews: int
    components: list[ComponentResult]
    strengths: list[str]
    warnings: list[str]
    missing: list[str]
    not_assessed: list[dict[str, str]]
    config_hash: str
    config: dict[str, Any]
    raw_score: float | None = None          # weighted average before any cap
    adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def component(self, key: str) -> ComponentResult | None:
        return next((c for c in self.components if c.key == key), None)


# ------------------------------------------------------------------ helpers
def _num(p: Mapping[str, Any], key: str) -> float | None:
    v = p.get(key)
    try:
        if v is None:
            return None
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _clip100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _result(key: str, cfg_w: dict[str, float], score: float | None, evidence: str,
            details: dict[str, Any] | None = None, missing: str | None = None) -> ComponentResult:
    return ComponentResult(
        key=key, label=COMPONENT_LABELS[key], weight=cfg_w[key],
        score=None if score is None else _clip100(score), available=score is not None,
        evidence=evidence, details=details or {}, missing_reason=missing,
    )


# ------------------------------------------------------------------ components
def _rating_quality(p, ctx, cfg, w) -> ComponentResult:
    n, mean = _num(p, "n_reviews") or 0, _num(p, "mean_rating")
    if n < 1 or mean is None:
        return _result("rating_quality", w, None, "", missing="No reviews available.")
    prior = float(ctx.get("global_mean_rating", 4.0))
    m = cfg.bayes_prior_weight
    bayes = (n * mean + m * prior) / (n + m)
    score = (bayes - 1.0) / 4.0 * 100.0
    return _result(
        "rating_quality", w, score,
        f"Mean {mean:.2f}★ over {int(n)} reviews; adjusted to {bayes:.2f}★ (prior {prior:.2f}★ x {m:g} reviews).",
        {"mean_rating": mean, "bayesian_rating": bayes, "n_reviews": int(n), "prior_mean": prior},
    )


def _rating_consistency(p, ctx, cfg, w) -> ComponentResult:
    n = _num(p, "n_reviews") or 0
    agree = _num(p, "rating_agreement_share")
    if n < 2 or agree is None:
        return _result("rating_consistency", w, None, "", missing="Needs at least 2 reviews.")
    subs: list[tuple[str, float, float, str]] = [
        ("agreement", 0.50, agree * 100.0, f"{agree:.0%} of ratings within 1★ of the mean"),
    ]
    mean, meta_avg = _num(p, "mean_rating"), _num(p, "meta_avg_rating")
    if n >= 10 and mean is not None and meta_avg is not None:
        diff = abs(mean - meta_avg)
        subs.append(("catalogue_alignment", 0.25, 100.0 * max(0.0, 1.0 - diff / 1.0),
                     f"sample mean {mean:.2f}★ vs catalogue {meta_avg:.2f}★"))
    gap = _num(p, "rating_trend_gap")
    if gap is not None:
        subs.append(("temporal_stability", 0.25, 100.0 * max(0.0, 1.0 - gap / 1.5),
                     f"early vs recent mean differs by {gap:.2f}★"))
    tw = sum(s[1] for s in subs)
    score = sum(s[1] * s[2] for s in subs) / tw
    return _result(
        "rating_consistency", w, score, "; ".join(s[3] for s in subs) + ".",
        {"sub_scores": {s[0]: round(s[2], 1) for s in subs}, "rating_std": _num(p, "rating_std")},
    )


def _review_volume(p, ctx, cfg, w) -> ComponentResult:
    n = _num(p, "n_reviews") or 0
    meta_n = _num(p, "meta_rating_number")
    if n < 1:
        return _result("review_volume", w, None, "", missing="No reviews available.")
    pop = max(n, meta_n) if meta_n is not None else n
    source = "catalogue rating count" if meta_n is not None and meta_n >= n else "analysed sample size"
    score = 100.0 * min(1.0, math.log1p(pop) / math.log1p(cfg.volume_target))
    return _result(
        "review_volume", w, score,
        f"{int(pop):,} ratings ({source}); full marks at {cfg.volume_target:g}. {int(n)} reviews were analysed.",
        {"effective_count": pop, "source": source, "n_analysed": int(n)},
    )


def _verified(p, ctx, cfg, w) -> ComponentResult:
    share, n = _num(p, "verified_share"), _num(p, "n_reviews") or 0
    if share is None or n < 1:
        return _result("verified_purchases", w, None, "", missing="Verified-purchase flag unavailable.")
    return _result("verified_purchases", w, share * 100.0,
                   f"{share:.0%} of {int(n)} analysed reviews are verified purchases.", {"verified_share": share})


def _sentiment(p, ctx, cfg, w) -> ComponentResult:
    n = int(_num(p, "n_sentiment") or 0)
    pos, neg, neu = _num(p, "sent_pos_share"), _num(p, "sent_neg_share"), _num(p, "sent_neu_share")
    if n < cfg.min_sentiment_reviews or pos is None or neg is None:
        why = ("Sentiment analysis was not run." if n == 0
               else f"Only {n} reviews have sentiment scores (need {cfg.min_sentiment_reviews}).")
        return _result("sentiment", w, None, "", missing=why)
    score = 50.0 * (1.0 + (pos - neg))
    return _result(
        "sentiment", w, score,
        f"{pos:.0%} positive, {(neu or 0):.0%} neutral, {neg:.0%} negative across {n} scored reviews.",
        {"positive": pos, "neutral": neu, "negative": neg, "n_scored": n, "sent_rating_corr": _num(p, "sent_rating_corr")},
    )


def _authenticity(p, ctx, cfg, w) -> ComponentResult:
    n = _num(p, "n_reviews") or 0
    sus, dup, mean_s = _num(p, "suspicious_share"), _num(p, "duplicate_share"), _num(p, "mean_suspicion")
    if n < 1 or sus is None or dup is None or mean_s is None:
        return _result("authenticity", w, None, "", missing="Suspicious-signal analysis unavailable.")
    burst = _num(p, "burst_ratio")
    index = (0.40 * min(1.0, sus / 0.30) + 0.25 * min(1.0, dup / 0.15)
             + 0.20 * (min(1.0, max(0.0, (burst - 0.20) / 0.50)) if burst is not None else 0.0)
             + 0.15 * mean_s)
    score = 100.0 * (1.0 - index)
    bits = [f"{sus:.0%} of reviews carry suspicious signals", f"{dup:.0%} repeat other reviews' text"]
    if burst is not None:
        bits.append(f"{burst:.0%} of reviews fall in the busiest 7-day window")
    return _result(
        "authenticity", w, score, "; ".join(bits) + ".",
        {"suspicious_share": sus, "duplicate_share": dup, "burst_ratio": burst, "mean_suspicion": mean_s,
         "n_flagged": int(_num(p, "n_suspicious") or 0), "index": index,
         "method": ctx.get("suspicion_method", "rules_only")},
    )


def _completeness(p, ctx, cfg, w) -> ComponentResult:
    if not bool(p.get("has_metadata", False)):
        return _result("listing_completeness", w, None, "",
                       missing="No catalogue metadata was found for this product in the sampled metadata.")
    from trust_assessor.features.product_features import COMPLETENESS_FIELDS

    present = [lbl for k, lbl in COMPLETENESS_FIELDS.items() if bool(p.get(k, False))]
    absent = [lbl for k, lbl in COMPLETENESS_FIELDS.items() if not bool(p.get(k, False))]
    score = 100.0 * len(present) / len(COMPLETENESS_FIELDS)
    txt = f"{len(present)} of {len(COMPLETENESS_FIELDS)} listing fields present"
    txt += f"; missing: {', '.join(absent)}." if absent else "."
    return _result("listing_completeness", w, score, txt, {"present": present, "absent": absent})


def _informativeness(p, ctx, cfg, w) -> ComponentResult:
    pct = _num(p, "helpful_prob_pct")
    if pct is None:
        return _result("informativeness", w, None, "",
                       missing="The review-helpfulness model has not been trained (run scripts/train_models.py).")
    mean_p = _num(p, "helpful_prob_mean")
    return _result("informativeness", w, pct * 100.0,
                   f"Predicted helpfulness {mean_p:.0%} on average, at the {pct:.0%} percentile of products.",
                   {"mean_predicted_helpfulness": mean_p, "percentile": pct})


_BUILDERS = {
    "rating_quality": _rating_quality, "rating_consistency": _rating_consistency,
    "review_volume": _review_volume, "verified_purchases": _verified, "sentiment": _sentiment,
    "authenticity": _authenticity, "listing_completeness": _completeness, "informativeness": _informativeness,
}


# ------------------------------------------------------------------ narrative
def _narrative(p, comps: list[ComponentResult], cfg: ScoringConfig) -> tuple[list[str], list[str], list[str]]:
    strengths, warnings, missing = [], [], []
    for c in comps:
        if not c.available:
            missing.append(f"{c.label}: {c.missing_reason}")
        elif c.weight > 0 and c.score >= cfg.strength_threshold:
            strengths.append(f"{c.label} is strong ({c.score:.0f}/100). {c.evidence}")
        elif c.weight > 0 and c.score < cfg.warning_threshold:
            warnings.append(f"{c.label} is weak ({c.score:.0f}/100). {c.evidence}")

    n = int(_num(p, "n_reviews") or 0)
    if n < 10:
        warnings.append(f"Only {n} reviews were analysed, so every statistic here has limited statistical support.")
    meta_n = _num(p, "meta_rating_number")
    if meta_n is not None and meta_n < 10:
        warnings.append(f"The catalogue lists only {int(meta_n)} ratings overall.")
    sus = _num(p, "suspicious_share")
    if sus is not None and sus >= 0.15 and n >= 10:
        warnings.append(f"{sus:.0%} of reviews carry suspicious signals. These warrant a closer look but are not proof of fake reviews.")
    burst = _num(p, "burst_ratio")
    if burst is not None and burst >= 0.40:
        warnings.append(f"{burst:.0%} of reviews were posted within a single 7-day window (review burst).")
    ver = _num(p, "verified_share")
    if ver is not None and ver < 0.50 and n >= 10:
        warnings.append(f"Fewer than half of reviews ({ver:.0%}) are verified purchases.")
    s5 = _num(p, "share_5star")
    if s5 is not None and s5 >= 0.90 and n >= 20:
        warnings.append(f"{s5:.0%} of reviews are 5★. This can reflect a genuinely excellent product or rating inflation; check the suspicious-signal analysis.")
    corr = _num(p, "sent_rating_corr")
    if corr is not None and corr < 0.20:
        warnings.append(f"Review text sentiment correlates only weakly with star ratings (ρ = {corr:.2f}), which can indicate rating/text inconsistency.")
    mean, meta_avg = _num(p, "mean_rating"), _num(p, "meta_avg_rating")
    if n >= 10 and mean is not None and meta_avg is not None and abs(mean - meta_avg) > 0.5:
        warnings.append(f"The sampled mean rating ({mean:.2f}★) differs from the catalogue average ({meta_avg:.2f}★) by more than 0.5★. The sample may be unrepresentative.")
    neg = _num(p, "sent_neg_share")
    if neg is not None and neg >= 0.30 and int(_num(p, "n_sentiment") or 0) >= cfg.min_sentiment_reviews:
        warnings.append(f"{neg:.0%} of reviews express negative sentiment.")
    if bool(p.get("has_metadata", False)) is False:
        missing.append("Product metadata (title, description, images, price) is not available in the sampled metadata.")
    else:
        if _num(p, "price") is None:
            missing.append("No listed price.")
    return strengths, warnings, missing


# ------------------------------------------------------------------ main entry
def assess_product(product: Mapping[str, Any], ctx: Mapping[str, Any], cfg: ScoringConfig) -> TrustAssessment:
    """Compute a TrustAssessment for one product row (dict-like)."""
    weights = cfg.normalised_weights()
    comps = [_BUILDERS[k](product, ctx, cfg, weights) for k in COMPONENT_KEYS]

    active_total = sum(c.weight for c in comps if c.weight > 0)
    avail_total = sum(c.weight for c in comps if c.available and c.weight > 0)
    coverage = (avail_total / active_total) if active_total > 0 else 0.0
    n = int(_num(product, "n_reviews") or 0)

    for c in comps:
        if c.available and avail_total > 0 and c.weight > 0:
            c.effective_weight = c.weight / avail_total
            c.contribution = c.effective_weight * (c.score or 0.0)

    strengths, warnings, missing = _narrative(product, comps, cfg)
    adjustments: list[str] = []
    raw_score: float | None = None

    if n < cfg.min_reviews_for_score or coverage < cfg.min_coverage or avail_total <= 0:
        score, raw_score, grade, confidence = None, None, "Insufficient evidence", "Very low"
        reason = (f"Fewer than {cfg.min_reviews_for_score} reviews" if n < cfg.min_reviews_for_score
                  else f"Evidence coverage {coverage:.0%} is below the {cfg.min_coverage:.0%} minimum")
        warnings.insert(0, f"No trust score was produced: {reason}.")
        for c in comps:
            c.effective_weight, c.contribution = 0.0, 0.0
    else:
        score = float(sum(c.contribution for c in comps))
        raw_score = score
        auth = next(c for c in comps if c.key == "authenticity")
        if (cfg.red_flag_cap_enabled and auth.available and auth.score < cfg.red_flag_authenticity_below
                and n >= cfg.medium_conf_reviews and score > cfg.red_flag_score_cap):
            adjustments.append(
                f"Red-flag cap: Review Authenticity Signals scored {auth.score:.0f} (< {cfg.red_flag_authenticity_below:g}) "
                f"on {n} reviews, so the score was capped at {cfg.red_flag_score_cap:g} (weighted average was {score:.1f}). "
                "Strong ratings cannot fully offset heavy suspicious-review signals."
            )
            score = cfg.red_flag_score_cap
        grade = "High trust" if score >= cfg.grade_high else "Moderate trust" if score >= cfg.grade_moderate else "Low trust"
        if n >= cfg.high_conf_reviews and coverage >= 0.90:
            confidence = "High"
        elif n >= cfg.medium_conf_reviews and coverage >= 0.70:
            confidence = "Medium"
        else:
            confidence = "Low"

    return TrustAssessment(
        parent_asin=str(product.get("parent_asin")), trust_score=None if score is None else round(score, 2),
        grade=grade, confidence=confidence, evidence_coverage=round(coverage, 4), n_reviews=n,
        components=comps, strengths=strengths, warnings=warnings, missing=missing,
        not_assessed=[{"item": a, "reason": b} for a, b in NOT_ASSESSED],
        config_hash=cfg.fingerprint(), config=cfg.to_dict(),
        raw_score=None if raw_score is None else round(raw_score, 2), adjustments=adjustments,
    )
