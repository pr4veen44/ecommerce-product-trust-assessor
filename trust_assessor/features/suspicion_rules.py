"""Transparent, rule-based suspicious-signal detection for reviews.

IMPORTANT: Amazon Reviews 2023 has **no ground-truth "fake review" labels**.
These rules therefore flag reviews that *warrant a closer look*; they never
declare a review fake. Each rule has an explicit weight and threshold that is
shown in the UI (Methodology / Model Insights pages).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (weight, human label, description with thresholds)
RULE_CATALOG: dict[str, dict] = {
    "exact_duplicate": {
        "weight": 0.30,
        "label": "Repeated text",
        "description": "Identical normalised text (>= 5 words) appears in at least one other review in the sample.",
    },
    "near_duplicate": {
        "weight": 0.25,
        "label": "Near-duplicate of another review of the same product",
        "description": "TF-IDF cosine similarity >= 0.90 with another review of the same product (>= 8 words).",
    },
    "user_same_day_burst": {
        "weight": 0.20,
        "label": "Reviewer posted many reviews on one day",
        "description": "The same reviewer posted >= 3 reviews on the same calendar day.",
    },
    "user_uniform_extreme": {
        "weight": 0.15,
        "label": "Reviewer only gives extreme ratings",
        "description": "Reviewer has >= 3 reviews in the sample and every rating is 1 or 5 stars.",
    },
    "product_day_burst": {
        "weight": 0.20,
        "label": "Review burst on a single day",
        "description": "This review is one of >= 5 reviews for the product on the same day, making up >= 20% of its reviews.",
    },
    "unverified_terse_extreme": {
        "weight": 0.15,
        "label": "Unverified, extreme rating, almost no text",
        "description": "Not a verified purchase, rating of 1 or 5 stars and <= 3 words of text.",
    },
    "sentiment_rating_mismatch": {
        "weight": 0.20,
        "label": "Text sentiment contradicts the star rating",
        "description": "Rating >= 4 with strongly negative text (P(neg) >= 0.90) or rating <= 2 with strongly positive text (P(pos) >= 0.90).",
    },
    "excessive_emphasis": {
        "weight": 0.15,
        "label": "Excessive emphasis with extreme sentiment",
        "description": "Heavy exclamation / capitalisation (>= 5% '!' or >= 50% capitals, >= 3 words) with sentiment confidence >= 0.99.",
    },
    "low_lexical_diversity": {
        "weight": 0.10,
        "label": "Very repetitive wording",
        "description": "Unique-word ratio < 0.40 in a review of >= 10 words.",
    },
}


def compute_rule_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sig_*`` boolean columns, ``rule_score`` (0-1) and ``rule_reasons``."""
    df = df.copy()
    wc = df["word_count"]
    have_sent = df["sent_pos"].notna() if "sent_pos" in df else pd.Series(False, index=df.index)
    p_pos = df["sent_pos"] if "sent_pos" in df else pd.Series(np.nan, index=df.index)
    p_neg = df["sent_neg"] if "sent_neg" in df else pd.Series(np.nan, index=df.index)
    conf = np.maximum(p_pos.fillna(0), p_neg.fillna(0))

    sig = {
        "exact_duplicate": (df["dup_text_global"] >= 1) & (wc >= 5),
        "near_duplicate": (df["max_sim_in_product"] >= 0.90) & (wc >= 8),
        "user_same_day_burst": df["user_same_day_count"] >= 3,
        "user_uniform_extreme": df["user_all_extreme"],
        "product_day_burst": (df["day_cluster_size"] >= 5) & (df["day_cluster_share"] >= 0.20),
        "unverified_terse_extreme": (~df["verified_purchase"]) & df["rating_extreme"] & (wc <= 3),
        "sentiment_rating_mismatch": have_sent & (
            ((df["rating"] >= 4) & (p_neg >= 0.90)) | ((df["rating"] <= 2) & (p_pos >= 0.90))
        ),
        "excessive_emphasis": have_sent & (wc >= 3) & (conf >= 0.99)
        & ((df["exclaim_ratio"] >= 0.05) | (df["caps_ratio"] >= 0.50)),
        "low_lexical_diversity": (wc >= 10) & (df["unique_word_ratio"] < 0.40),
    }

    score = np.zeros(len(df))
    labels: list[pd.Series] = []
    for name, mask in sig.items():
        mask = mask.fillna(False).astype(bool)
        df[f"sig_{name}"] = mask
        score += mask.to_numpy() * RULE_CATALOG[name]["weight"]
        labels.append(mask.map({True: RULE_CATALOG[name]["label"], False: ""}))
    df["rule_score"] = np.minimum(score, 1.0)
    df["rule_reasons"] = [
        "; ".join(x for x in row if x) for row in zip(*[s.tolist() for s in labels])
    ]
    return df


def triggered_rules(row: pd.Series) -> list[dict]:
    """Rule-based explanation for a single review row."""
    out = []
    for name, meta in RULE_CATALOG.items():
        if bool(row.get(f"sig_{name}", False)):
            out.append({"rule": name, "label": meta["label"], "weight": meta["weight"],
                        "description": meta["description"]})
    return out
