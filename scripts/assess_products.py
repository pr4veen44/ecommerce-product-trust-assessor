#!/usr/bin/env python
"""Load products into SQLite and compute a Trust Score assessment for every product.

Re-run after changing scoring weights (data/scoring_config.json) or retraining models.
Each run appends new assessments, so history is preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trust_assessor.config import get_settings  # noqa: E402
from trust_assessor.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    s = get_settings()
    setup_logging(s.log_level)
    from trust_assessor.services.assessment_service import assess_all

    if not s.products_path.exists():
        print("No processed data found. Run: python scripts/prepare_dataset.py")
        return 1
    res = assess_all(s, source="batch")
    print(f"\nAssessed {res['n_products']:,} products ({res['n_scored']:,} received a score, "
          f"mean {res['mean_score']:.1f}). Run id: {res['run_id']}")
    print("Next: python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
