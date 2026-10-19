"""Reusable NiceGUI building blocks."""

from __future__ import annotations

import math
from typing import Any, Iterable

from nicegui import ui

from trust_assessor.config import get_settings
from trust_assessor.ui import theme


def card(classes: str = "") -> ui.element:
    return ui.element("div").classes(f"tc-card w-full {classes}")


def heading(text: str, sub: str | None = None) -> None:
    ui.label(text).classes("tc-h2")
    if sub:
        ui.label(sub).classes("tc-sub")


def kpi(label: str, value: str, icon: str = "insights", hint: str | None = None, color: str = "#4f46e5") -> None:
    with card("!p-4"):
        with ui.row().classes("items-center justify-between w-full no-wrap"):
            with ui.column().classes("gap-1"):
                ui.label(label).classes("tc-kpi-label")
                ui.label(value).classes("tc-kpi-value")
                if hint:
                    ui.label(hint).classes("tc-sub")
            ui.icon(icon, size="28px").style(f"color:{color}; opacity:.85")


def grade_badge(grade: str | None) -> ui.badge:
    return ui.badge(grade or "Not assessed", color=theme.GRADE_QUASAR.get(grade or "", "grey")).classes("text-sm px-2 py-1")


def confidence_chip(confidence: str | None) -> None:
    icon = theme.CONFIDENCE_ICON.get(confidence or "", "help_outline")
    with ui.row().classes("items-center gap-1 no-wrap"):
        ui.icon(icon, size="18px").classes("text-slate-500")
        ui.label(f"{confidence or 'n/a'} confidence").classes("text-sm text-slate-600")


def bullet_list(items: Iterable[str], icon: str, color: str, empty: str = "Nothing to report.") -> None:
    items = list(items)
    if not items:
        ui.label(empty).classes("tc-sub")
        return
    with ui.column().classes("gap-2 w-full"):
        for text in items:
            with ui.element("div").classes("tc-li"):
                ui.icon(icon, size="18px").style(f"color:{color}; margin-top:1px; flex:none")
                ui.label(text)


def info_note(text: str, icon: str = "info", color: str = "blue") -> None:
    with ui.row().classes(f"w-full items-start no-wrap gap-2 p-3 rounded-lg bg-{color}-50 border border-{color}-200"):
        ui.icon(icon, size="20px").classes(f"text-{color}-700 mt-0.5")
        ui.label(text).classes(f"text-sm text-{color}-900")


def command_block(commands: list[str]) -> None:
    ui.code("\n".join(commands), language="bash").classes("w-full")


def setup_required() -> None:
    with card("max-w-3xl mx-auto"):
        with ui.column().classes("gap-3"):
            ui.icon("inventory_2", size="40px").classes("text-indigo-500")
            ui.label("No processed data found yet").classes("tc-h1")
            ui.label("The app analyses a sample of the Amazon Reviews 2023 dataset. Prepare it once, train the models, "
                     "then (re)start the app. Everything runs locally.").classes("tc-sub")
            command_block([
                "python scripts/prepare_dataset.py        # download + clean a sample, extract features, run sentiment",
                "python scripts/train_models.py           # anomaly + helpfulness models, SHAP",
                "python scripts/assess_products.py        # compute trust scores into SQLite",
                "python main.py                           # launch this dashboard",
            ])
            ui.label("Tip: add  --sentiment lexicon  to the first command for a quick run without PyTorch.").classes("tc-sub")


def clean_value(v: Any) -> Any:
    """Make values JSON/Vue friendly (NaN -> None, numpy -> python)."""
    try:
        import numpy as np

        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is None:
        return None
    try:
        import pandas as pd

        if v is pd.NA or v is pd.NaT:
            return None
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
    except ImportError:  # pragma: no cover
        pass
    return v


def product_thumb(url: str | None, size: str = "w-20 h-20") -> None:
    if get_settings().show_product_images and url:
        ui.image(url).classes(f"{size} rounded-lg object-contain bg-white border").props("loading=lazy no-spinner")
    else:
        with ui.element("div").classes(f"{size} rounded-lg bg-slate-100 flex items-center justify-center"):
            ui.icon("inventory_2", size="28px").classes("text-slate-400")


SCORE_CELL = """
<q-td :props="props">
  <div v-if="props.row.trust_score !== null" class="row items-center no-wrap" style="gap:8px">
    <q-linear-progress :value="props.row.trust_score/100" :color="props.row.grade_q" size="8px" rounded style="width:64px"/>
    <b>{{ props.row.trust_score.toFixed(1) }}</b>
  </div>
  <span v-else class="text-grey">n/a</span>
</q-td>
"""

GRADE_CELL = """
<q-td :props="props"><q-badge :color="props.row.grade_q" :label="props.row.grade || 'n/a'" /></q-td>
"""

TITLE_CELL = """
<q-td :props="props" style="max-width: 420px; white-space: normal">
  <div class="tc-clamp2" style="font-weight:550">{{ props.row.title }}</div>
  <div class="text-caption text-grey-7 tc-mono">{{ props.row.parent_asin }}</div>
</q-td>
"""


def product_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        d = {k: clean_value(v) for k, v in r.items() if k not in ("components",)}
        d["grade_q"] = theme.GRADE_QUASAR.get(d.get("grade") or "", "grey")
        d["coverage_pct"] = None if d.get("evidence_coverage") is None else round(d["evidence_coverage"] * 100)
        d["mean_rating_txt"] = "n/a" if d.get("mean_rating") is None else f"{d['mean_rating']:.2f}"
        out.append(d)
    return out


PRODUCT_COLUMNS = [
    {"name": "title", "label": "Product", "field": "title", "align": "left", "sortable": True},
    {"name": "trust_score", "label": "Trust score", "field": "trust_score", "align": "left", "sortable": True},
    {"name": "grade", "label": "Grade", "field": "grade", "align": "left"},
    {"name": "confidence", "label": "Confidence", "field": "confidence", "align": "left"},
    {"name": "n_reviews", "label": "Reviews", "field": "n_reviews", "align": "right", "sortable": True},
    {"name": "mean_rating_txt", "label": "Avg ★", "field": "mean_rating_txt", "align": "right"},
    {"name": "coverage_pct", "label": "Evidence", "field": "coverage_pct", "align": "right"},
]


def product_table(rows: list[dict[str, Any]], selection: str | None = None, page_size: int = 15) -> ui.table:
    table = ui.table(columns=PRODUCT_COLUMNS, rows=product_table_rows(rows), row_key="parent_asin",
                     selection=selection, pagination={"rowsPerPage": page_size}).classes("w-full").props("flat dense")
    table.add_slot("body-cell-title", TITLE_CELL)
    table.add_slot("body-cell-trust_score", SCORE_CELL)
    table.add_slot("body-cell-grade", GRADE_CELL)
    table.on("row-click", lambda e: ui.navigate.to(f"/product/{e.args[1]['parent_asin']}"))
    return table
