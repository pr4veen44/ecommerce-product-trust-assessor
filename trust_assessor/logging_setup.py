"""Logging helper shared by scripts and the web app."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Keep noisy third-party loggers quiet.
    for noisy in ("httpx", "httpcore", "urllib3", "fsspec", "huggingface_hub", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
