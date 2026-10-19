"""Configurable parameters of the Trust Score (weights, thresholds, priors)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

COMPONENT_KEYS = [
    "rating_quality", "rating_consistency", "review_volume", "verified_purchases",
    "sentiment", "authenticity", "listing_completeness", "informativeness",
]

COMPONENT_LABELS = {
    "rating_quality": "Rating Quality",
    "rating_consistency": "Rating Consistency",
    "review_volume": "Review Volume",
    "verified_purchases": "Verified Purchases",
    "sentiment": "Review Sentiment",
    "authenticity": "Review Authenticity Signals",
    "listing_completeness": "Listing Completeness",
    "informativeness": "Review Informativeness (ML)",
}

COMPONENT_DESCRIPTIONS = {
    "rating_quality": "Bayesian-adjusted average star rating, shrunk toward the dataset mean when there are few reviews.",
    "rating_consistency": "How tightly reviews agree with each other, with the catalogue average rating, and over time.",
    "review_volume": "Log-scaled amount of review evidence (uses the catalogue rating count when available).",
    "verified_purchases": "Share of analysed reviews that are verified purchases.",
    "sentiment": "Net positive sentiment (positive share minus negative share) of review text.",
    "authenticity": "Absence of suspicious signals: repeated text, near-duplicates, review bursts, anomaly flags. Not a fake-review detector.",
    "listing_completeness": "Share of expected listing fields present in the product metadata.",
    "informativeness": "Percentile of the product's mean predicted review helpfulness (XGBoost). A proxy for review usefulness.",
}

DEFAULT_WEIGHTS = {
    "rating_quality": 0.22,
    "rating_consistency": 0.12,
    "review_volume": 0.12,
    "verified_purchases": 0.12,
    "sentiment": 0.12,
    "authenticity": 0.18,
    "listing_completeness": 0.08,
    "informativeness": 0.04,
}


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    bayes_prior_weight: float = 5.0          # pseudo-reviews pulling the mean toward the dataset average
    volume_target: float = 200.0             # rating count that earns a full volume score
    min_sentiment_reviews: int = 3
    min_reviews_for_score: int = 3           # below this no score is produced
    min_coverage: float = 0.50               # minimum share of total weight that must be available
    grade_high: float = 75.0
    grade_moderate: float = 55.0
    strength_threshold: float = 80.0
    warning_threshold: float = 50.0
    red_flag_cap_enabled: bool = True        # cap the score when authenticity signals are very poor
    red_flag_authenticity_below: float = 30.0
    red_flag_score_cap: float = 54.9         # just below the 'Moderate trust' threshold
    high_conf_reviews: int = 30
    medium_conf_reviews: int = 10

    def normalised_weights(self) -> dict[str, float]:
        total = sum(max(0.0, w) for w in self.weights.values()) or 1.0
        return {k: max(0.0, self.weights.get(k, 0.0)) / total for k in COMPONENT_KEYS}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in known and k != "weights"})
        cfg.weights = {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in data.get("weights", {}).items() if k in DEFAULT_WEIGHTS}}
        return cfg

    @classmethod
    def load(cls, path: Path) -> "ScoringConfig":
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                pass
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
