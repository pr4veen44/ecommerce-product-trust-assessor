#!/usr/bin/env python
"""Train the ML models on the prepared dataset.

* IsolationForest (unsupervised) -> review anomaly score; combined with rule signals into a suspicion score
* XGBoost (supervised)           -> review helpfulness (label: helpful_vote >= 1), evaluated on held-out products
* SHAP                           -> global feature attributions saved for the dashboard
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
    from trust_assessor.services.training import train_all

    manifest = train_all(s)
    h = manifest["helpfulness_model"]
    if h.get("trained"):
        print(f"\nHelpfulness model: test ROC-AUC {h['roc_auc_test']:.3f}, PR-AUC {h['pr_auc_test']:.3f}")
    else:
        print(f"\nHelpfulness model was NOT trained: {h.get('reason')}")
    print(f"Model version: {manifest['model_version']}")
    print("Next: python scripts/assess_products.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
