#!/usr/bin/env python
"""Download a configurable sample of one Amazon Reviews 2023 category and build processed Parquet files.

Examples
--------
    python scripts/prepare_dataset.py                                   # defaults from .env
    python scripts/prepare_dataset.py --category Appliances --sample 30000
    python scripts/prepare_dataset.py --sentiment lexicon               # no torch needed
    python scripts/prepare_dataset.py --list-categories
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trust_assessor.config import get_settings  # noqa: E402
from trust_assessor.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default=s.category, help="Amazon category file name, e.g. All_Beauty")
    ap.add_argument("--sample", type=int, default=s.sample_reviews, help="number of reviews to sample")
    ap.add_argument("--scan-multiplier", type=int, default=s.scan_multiplier,
                    help="read N x sample lines, then randomly subsample (less head-of-file bias)")
    ap.add_argument("--min-reviews", type=int, default=s.min_reviews_per_product)
    ap.add_argument("--sentiment", choices=["transformers", "lexicon", "none"], default=s.sentiment_backend)
    ap.add_argument("--sentiment-max-reviews", type=int, default=s.sentiment_max_reviews,
                    help="score only a random subset of reviews (0 = all)")
    ap.add_argument("--seed", type=int, default=s.seed)
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw samples and re-download")
    ap.add_argument("--list-categories", action="store_true")
    args = ap.parse_args()

    setup_logging(s.log_level)
    s = replace(
        s, category=args.category, sample_reviews=args.sample, scan_multiplier=max(1, args.scan_multiplier),
        min_reviews_per_product=args.min_reviews, sentiment_backend=args.sentiment,
        sentiment_max_reviews=args.sentiment_max_reviews, seed=args.seed,
    )

    if args.list_categories:
        from trust_assessor.data.ingestion import available_categories

        print("\n".join(available_categories(s)))
        return 0

    from trust_assessor.services.pipeline import prepare_dataset

    manifest = prepare_dataset(s, refresh=args.refresh)
    st = manifest["stats"]
    print(f"\nDone. {st['n_reviews']:,} reviews, {st['n_products']:,} products "
          f"({st['n_products_with_metadata']:,} with metadata).")
    print(f"Dataset version: {manifest['dataset_version']}")
    print("Next: python scripts/train_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
