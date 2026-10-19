#!/usr/bin/env python
"""Run prepare -> train -> assess in one go (equivalent to the three individual scripts)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    passthrough = sys.argv[1:]          # forwarded to prepare_dataset.py, e.g. --sample 20000 --sentiment lexicon
    steps = [["prepare_dataset.py", *passthrough], ["train_models.py"], ["assess_products.py"]]
    for step in steps:
        print(f"\n=== {' '.join(step)} ===")
        code = subprocess.call([sys.executable, str(ROOT / "scripts" / step[0]), *step[1:]], cwd=ROOT)
        if code:
            return code
    print("\nAll done. Launch the app with: python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
