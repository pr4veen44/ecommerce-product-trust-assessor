"""Colours and small formatting helpers shared by all pages."""

from __future__ import annotations

import math
from typing import Any

PRIMARY = "#4f46e5"
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"

GRADE_COLORS = {
    "High trust": "#16a34a",
    "Moderate trust": "#d97706",
    "Low trust": "#dc2626",
    "Insufficient evidence": "#94a3b8",
}
GRADE_QUASAR = {"High trust": "positive", "Moderate trust": "warning", "Low trust": "negative",
                "Insufficient evidence": "grey"}
CONFIDENCE_ICON = {"High": "verified", "Medium": "rule", "Low": "warning_amber", "Very low": "error_outline"}
SERIES_COLORS = ["#4f46e5", "#0ea5e9", "#f59e0b", "#10b981", "#ec4899", "#8b5cf6"]

CSS = """
:root { --brand: #4f46e5; }
body { background: #f8fafc; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.tc-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.tc-kpi-value { font-size: 1.75rem; font-weight: 700; line-height: 1.1; color: #0f172a; }
.tc-kpi-label { font-size: .75rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: #64748b; }
.tc-h1 { font-size: 1.6rem; font-weight: 750; color: #0f172a; line-height: 1.2; }
.tc-h2 { font-size: 1.05rem; font-weight: 650; color: #0f172a; }
.tc-sub { color: #64748b; font-size: .875rem; }
.tc-clamp2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.tc-li { display: flex; gap: 8px; align-items: flex-start; font-size: .875rem; line-height: 1.35; color: #1e293b; }
.q-table tbody tr { cursor: pointer; }
.q-table__card { box-shadow: none !important; }
.tc-nav a { border-radius: 10px; }
.tc-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; }
.nicegui-content { padding: 0 !important; }
"""


def fmt_score(x: Any, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def fmt_pct(x: Any, digits: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def fmt_price(x: Any) -> str:
    return "not listed" if x is None or (isinstance(x, float) and math.isnan(x)) else f"${x:,.2f}"


def fmt_int(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{int(x):,}"


def grade_color(grade: str | None) -> str:
    return GRADE_COLORS.get(grade or "", "#94a3b8")


def score_color(score: float | None, high: float = 75.0, moderate: float = 55.0) -> str:
    if score is None:
        return GRADE_COLORS["Insufficient evidence"]
    return GRADE_COLORS["High trust"] if score >= high else GRADE_COLORS["Moderate trust"] if score >= moderate else GRADE_COLORS["Low trust"]


def short(text: str | None, n: int = 90) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
